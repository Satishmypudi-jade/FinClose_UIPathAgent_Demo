import pandas as pd
from datetime import datetime


def last_bus_day_countdown(first_bus_day, last_bus_day):
    formatted_first_bd = first_bus_day.strftime('%d-%b-%Y')
    formatted_last_bd = last_bus_day.strftime('%d-%b-%Y')
    # Convert current timestamp and last business day to datetime objects
    date_obj = datetime.combine(last_bus_day, datetime.min.time())
    curr_Timestamp = pd.Timestamp.now()
    countDown = date_obj - curr_Timestamp.to_pydatetime()
    cd_days = countDown.days
    cd_total_seconds = countDown.seconds
    cd_hours = cd_total_seconds // 3600
    cd_minutes = (cd_total_seconds % 3600) // 60
    cd_seconds = cd_total_seconds % 60
    return formatted_first_bd, formatted_last_bd, cd_days, cd_hours, cd_minutes, cd_seconds
