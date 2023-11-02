from datetime import datetime
from io import BytesIO
from zipfile import ZipFile
from typing import Optional
from tqdm import tqdm

# import plotly.express as px

import polars as pl
import requests

base_url = "http://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/{year:04d}/MMSDM_{year:04d}_{month:02d}/MMSDM_Historical_Data_SQLLoader/DATA/PUBLIC_DVD_{table}_{year:04d}{month:02d}010000.zip"


def extract_columns(zf: ZipFile) -> set[str]:
    # Takes the zipfile and returns it as a dataframe
    return set(
        pl.read_csv(
            zf.open(str(zf.filelist[0].filename)).read(),
            n_rows=2,
            skip_rows=1,
        ).columns
    )


def get_table(
    table: str,
    dt: datetime,
    drop_columns: Optional[set[str]] = None,
    datetime_columns: Optional[set[str]] = None,
    string_columns: Optional[set[str]] = None,
    integer_columns: Optional[set[str]] = None,
    non_nullable_columns: Optional[set[str]] = None,
) -> requests.Response:
    
    # Gets the web url and returns it as a dataframe
    # This is a generic function that's employed by all the other getter functions for forecast horizons

    response = requests.get(base_url.format(table=table, year=dt.year, month=dt.month))
    response.raise_for_status()

    # Default to empty set
    drop_columns = drop_columns or set()
    string_columns = string_columns or set()
    datetime_columns = datetime_columns or set()
    integer_columns = integer_columns or set()

    data = BytesIO(response.content)

    with ZipFile(data) as zf:
        columns = extract_columns(zf)
        dtypes = {
            **{col: pl.Float32 for col in columns - drop_columns - datetime_columns},
            **{col: pl.Datetime for col in datetime_columns},
            **{col: pl.Utf8 for col in string_columns},
            **{col: pl.Int64 for col in integer_columns},
        }
        return (
            pl.concat(
                [
                    pl.read_csv(
                        zf.open(file.filename).read(),
                        skip_rows=1,
                        dtypes=dtypes,
                    ).lazy()
                    for file in zf.filelist
                ]
            )
            .drop_nulls(subset=non_nullable_columns)
            .drop(columns=list(drop_columns))
        )


def select_column_subset(df: pl.LazyFrame) -> pl.LazyFrame:

    # Just keep the necessary columns

    return df.select(
        [
            "DATETIME",
            "forecast_type",
            "forecast_at",
            "forecast_horizon_mins",
            "REGIONID",
            "TOTALDEMAND",
        ]
    )


def get_p5min_regionsolution(dt: datetime) -> pl.LazyFrame:
    """
    Gather 5-minutely forecasts from NEMweb.
    These forecasts are up to one hour ahead of the forecast time.
    """
    drop_columns = {"I", "P5MIN", "REGIONSOLUTION", "7"}
    string_columns = {"I", "P5MIN", "REGIONSOLUTION", "REGIONID"}
    datetime_columns = {"RUN_DATETIME", "INTERVAL_DATETIME", "LASTCHANGED"}
    integer_columns = {"INTERVENTION"}
    non_nullable_columns = {"RUN_DATETIME", "INTERVAL_DATETIME", "REGIONID"}

    df = select_column_subset(
        get_table(
            table="P5MIN_REGIONSOLUTION",
            dt=dt,
            drop_columns=drop_columns,
            string_columns=string_columns,
            datetime_columns=datetime_columns,
            integer_columns=integer_columns,
            non_nullable_columns=non_nullable_columns,
        )
        # This is where the NEMweb data will be converted to a more usable format
        # If you want any additional columns, here's where to grab them!
        # Look on the NEMwebsite for more information
        .with_columns(
            [
                (pl.col("INTERVAL_DATETIME") - pl.col("RUN_DATETIME"))
                .dt.cast_time_unit("ms")
                .dt.minutes()
                .alias("forecast_horizon_mins"),
                pl.lit("P5MIN").alias("forecast_type"),
            ]
        ).rename({"INTERVAL_DATETIME": "DATETIME", "RUN_DATETIME": "forecast_at"})
    )

    return df


def get_predispatch(dt: datetime) -> pl.LazyFrame:
    """
    Gather half-hourly forecasts from NEMweb.
    These forecasts are up to 2 days ahead of the forecast time.

    More information can be found here:
    https://nemweb.com.au/Reports/Current/MMSDataModelReport/Electricity/MMS%20Data%20Model%20Report_files/MMS_285.htm
    """
    drop_columns = {"I", "PREDISPATCH", "REGION_SOLUTION", "5", "RUNNO"}
    string_columns = {"I", "PREDISPATCH", "REGION_SOLUTION", "REGIONID"}
    datetime_columns = {"DATETIME", "LASTCHANGED"}
    integer_columns = {"INTERVENTION"}
    non_nullable_columns = {"DATETIME", "REGIONID"}

    df = select_column_subset(
        get_table(
            table="PREDISPATCHREGIONSUM_D",
            dt=dt,
            drop_columns=drop_columns,
            string_columns=string_columns,
            datetime_columns=datetime_columns,
            integer_columns=integer_columns,
            non_nullable_columns=non_nullable_columns,
        )
        # This is where the NEMweb data will be converted to a more usable format
        # If you want any additional columns, here's where to grab them!
        .with_columns(
            [
                pl.lit("PREDISPATCH").alias("forecast_type"),
                pl.col("LASTCHANGED").dt.round("30m").alias("forecast_at"),
            ]
        ).with_columns(
            [
                (pl.col("DATETIME") - pl.col("forecast_at"))
                .dt.cast_time_unit("ms")
                .dt.minutes()
                .alias("forecast_horizon_mins"),
            ]
        )
    )

    return df


def get_pdpasa(dt: datetime) -> pl.LazyFrame:
    """
    Get half-hourly forecasts up to 2 days ahead (next trading interval until the end of the next trading day).

    More information can be found here:
    https://nemweb.com.au/Reports/Current/MMSDataModelReport/Electricity/MMS%20Data%20Model%20Report_files/MMS_485.htm#1
    https://aemo.com.au/en/energy-systems/electricity/national-electricity-market-nem/nem-forecasting-and-planning/forecasting-and-reliability/projected-assessment-of-system-adequacy

    """
    drop_columns = {"I", "PDPASA", "REGIONSOLUTION", "6", "RUNNO", "RUNTYPE"}
    string_columns = {"I", "PDPASA", "REGIONSOLUTION", "REGIONID"}
    datetime_columns = {"RUN_DATETIME", "INTERVAL_DATETIME", "LASTCHANGED"}
    non_nullable_columns = {"RUN_DATETIME", "INTERVAL_DATETIME", "REGIONID"}

    df = select_column_subset(
        get_table(
            table="PDPASA_REGIONSOLUTION",
            dt=dt,
            drop_columns=drop_columns,
            string_columns=string_columns,
            datetime_columns=datetime_columns,
            non_nullable_columns=non_nullable_columns,
        )
        # This is where the NEMweb data will be converted to a more usable format
        # If you want any additional columns, here's where to grab them!
        .with_columns(
            [
                (pl.col("INTERVAL_DATETIME") - pl.col("RUN_DATETIME"))
                .dt.cast_time_unit("ms")
                .dt.minutes()
                .alias("forecast_horizon_mins"),
                pl.lit("PDPASA").alias("forecast_type"),
            ]
        )
        # Note: I'm taking the 50th percentile (i.e. median) forecast as the total demand forecast
        .rename(
            {
                "INTERVAL_DATETIME": "DATETIME",
                "RUN_DATETIME": "forecast_at",
                "DEMAND50": "TOTALDEMAND",
            }
        )
    )

    return df


def get_stpasa(dt: datetime) -> pl.LazyFrame:
    """
    Get half-hourly forecasts up to 6 days ahead.
    stpasa is the one we'll have to use for the 1 week prediction. That's going to be our best comparison

    More information can be found here:
    https://nemweb.com.au/Reports/Current/MMSDataModelReport/Electricity/MMS%20Data%20Model%20Report_files/MMS_353.htm#1
    https://aemo.com.au/en/energy-systems/electricity/national-electricity-market-nem/nem-forecasting-and-planning/forecasting-and-reliability/projected-assessment-of-system-adequacy

    """
    drop_columns = {"I", "STPASA", "REGIONSOLUTION", "6", "RUNNO", "RUNTYPE"}
    string_columns = {"I", "STPASA", "REGIONSOLUTION", "REGIONID"}
    datetime_columns = {"RUN_DATETIME", "INTERVAL_DATETIME", "LASTCHANGED"}
    non_nullable_columns = {"RUN_DATETIME", "INTERVAL_DATETIME", "REGIONID"}

    df = select_column_subset(
        get_table(
            table="STPASA_REGIONSOLUTION",
            dt=dt,
            drop_columns=drop_columns,
            string_columns=string_columns,
            datetime_columns=datetime_columns,
            non_nullable_columns=non_nullable_columns,
        )  # This is where the NEMweb data will be converted to a more usable format
        # If you want any additional columns, here's where to grab them!
        .with_columns(
            [
                (pl.col("INTERVAL_DATETIME") - pl.col("RUN_DATETIME"))
                .dt.cast_time_unit("ms")
                .dt.minutes()
                .alias("forecast_horizon_mins"),
                pl.lit("STPASA").alias("forecast_type"),
            ]
        )
        # Note: I'm taking the 50th percentile (i.e. median) forecast as the total demand forecast
        .rename(
            {
                "INTERVAL_DATETIME": "DATETIME",
                "RUN_DATETIME": "forecast_at",
                "DEMAND50": "TOTALDEMAND",
            }
        )
    )

    return df


def get_dispatch(dt: datetime) -> pl.LazyFrame:
    """
    Get dispatch data from NEMweb. This is the actual dispatch data, not the forecasted data.
    """

    drop_columns = {"I", "DISPATCH", "REGIONSUM", "5", "RUNNO"}
    string_columns = {"I", "DISPATCH", "REGIONSUM", "REGIONID"}
    datetime_columns = {"SETTLEMENTDATE", "LASTCHANGED"}
    integer_columns = {"INTERVENTION"}
    non_nullable_columns = {"SETTLEMENTDATE", "REGIONID"}

    df = (
        get_table(
            table="DISPATCHREGIONSUM",
            dt=dt,
            drop_columns=drop_columns,
            string_columns=string_columns,
            datetime_columns=datetime_columns,
            integer_columns=integer_columns,
            non_nullable_columns=non_nullable_columns,
        )
        # This is where the NEMweb data will be converted to a more usable format
        # If you want any additional columns, here's where to grab them!
        .with_columns(
            [
                pl.lit("DISPATCH").alias("forecast_type"),
                pl.lit(None, dtype=pl.Datetime).alias("forecast_at"),
                pl.lit(None, dtype=pl.Int64).alias("forecast_horizon_mins"),
            ]
        )
        .rename({"SETTLEMENTDATE": "DATETIME"})
        .select(
            [
                "DATETIME",
                "forecast_type",
                "forecast_at",
                "forecast_horizon_mins",
                "REGIONID",
                "TOTALDEMAND",
            ]
        )
    )

    return df


def main():
    start = datetime(2022, 3, 1)
    end = datetime(2022, 5, 1)
    dfs: list[pl.LazyFrame] = []

    date_range = pl.datetime_range(start, end, "1mo", eager=True)

    for dt in tqdm(date_range, desc="Fetching AEMO data", total=len(date_range)):
        # Combine dispatch and forecasts
        get_functions = [
            get_dispatch,
            get_p5min_regionsolution,
            get_predispatch,
            get_pdpasa,
            get_stpasa,
        ]
        df = pl.concat(
            [
                get(dt)
                for get in tqdm(
                    get_functions,
                    total=len(get_functions),
                    desc=f"Fetching data for {dt.strftime('%Y-%m')}",
                )
            ]
        )
        dfs.append(df)
        # break

        # Plot -- left commented out as you likely want to reduce the amount of data you're plotting
        # for the snippet below, I've reduced it to 3 days of data and only half-hourly forecasts (or dispatch)
        # fig = px.line(
        #     df.filter(pl.col("DATETIME") < datetime(2022, 1, 4))
        #     .filter(
        #         (pl.col("forecast_horizon_mins") == 30)
        #         | (pl.col("forecast_horizon_mins").is_null())
        #     )
        #     .sort(pl.col("DATETIME"))
        #     .to_pandas(),
        #     x="DATETIME",
        #     y="TOTALDEMAND",
        #     color="forecast_type",
        #     facet_row="REGIONID",
        #     facet_col="forecast_horizon_mins",
        # )
        # fig.show()

        df = (
            pl.concat(dfs)
            .filter(pl.col("DATETIME").is_between(start, end))
            .sink_parquet(
                f"AEMO_load_data_{start.strftime('%Y-%m')}_{end.strftime('%Y-%m')}_{dt}.parquet"
            )
        )
    print("Done!")


if __name__ == "__main__":
    main()