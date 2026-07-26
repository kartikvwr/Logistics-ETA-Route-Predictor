import pandas as pd
import numpy as np


class FeatureEngineer:

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculate distance (km) between two GPS coordinates.
        """

        R = 6371

        lat1 = np.radians(lat1)
        lon1 = np.radians(lon1)
        lat2 = np.radians(lat2)
        lon2 = np.radians(lon2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1)
            * np.cos(lat2)
            * np.sin(dlon / 2) ** 2
        )

        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def create_features(self, df):

        df = df.copy()

        # --------------------------
        # Distance
        # --------------------------

        df["Distance_km"] = self.haversine_distance(
            df["Store_Latitude"],
            df["Store_Longitude"],
            df["Drop_Latitude"],
            df["Drop_Longitude"],
        )

        # --------------------------
        # Date
        # --------------------------

        df["Order_Date"] = pd.to_datetime(df["Order_Date"])

        df["Day"] = df["Order_Date"].dt.day
        df["Month"] = df["Order_Date"].dt.month
        df["Weekday"] = df["Order_Date"].dt.dayofweek

        # --------------------------
        # Time
        # --------------------------

        df["Order_Time"] = pd.to_datetime(
            df["Order_Time"],
            format="%H:%M:%S"
        )

        df["Pickup_Time"] = pd.to_datetime(
            df["Pickup_Time"],
            format="%H:%M:%S"
        )

        df["Order_Hour"] = df["Order_Time"].dt.hour

        df["Pickup_Delay_Min"] = (
            (
                df["Pickup_Time"]
                - df["Order_Time"]
            ).dt.total_seconds()
            / 60
        )

        # --------------------------
        # Remove unnecessary columns
        # --------------------------

        df.drop(
            columns=[
                "Order_ID",
                "Store_Latitude",
                "Store_Longitude",
                "Drop_Latitude",
                "Drop_Longitude",
                "Order_Date",
                "Order_Time",
                "Pickup_Time",
            ],
            inplace=True,
        )

        return df


if __name__ == "__main__":

    from data_loader import DataLoader

    loader = DataLoader("data/raw/delivery_data.csv")

    data = loader.load_data()

    data = loader.clean_data(data)

    engineer = FeatureEngineer()

    processed = engineer.create_features(data)

    print(processed.head())

    print("\nColumns:\n")

    print(processed.columns)