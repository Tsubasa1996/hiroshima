import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import pickle



class ScoringService(object):

    @classmethod
    def get_model(cls, model_path):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.

        Returns:
            bool: The return value. True for success.
        """
        # アンサンブルモデルか
        cls.emsemble = True
        # 雨量データを使用するか
        cls.userain = True


        # モデルのパス
        # 潮位ラベル=1

        if cls.emsemble:
            cls.L_models1 = []
            # 使用するモデルを順に読み込む
            cls.L_models1.append(pickle.load(open(os.path.join(model_path, "model_20230102-2-7.pkl"), 'rb')))
            cls.L_models1.append(pickle.load(open(os.path.join(model_path, "model_20230102-2-7-fold0.pkl"), 'rb')))
            cls.L_models1.append(pickle.load(open(os.path.join(model_path, "model_20230102-2-7-fold1.pkl"), 'rb')))
            cls.L_models1.append(pickle.load(open(os.path.join(model_path, "model_20230102-2-7-fold2.pkl"), 'rb')))
            cls.L_models1.append(pickle.load(open(os.path.join(model_path, "model_20230102-2-7-fold3.pkl"), 'rb')))
            cls.L_models1.append(pickle.load(open(os.path.join(model_path, "model_20230102-2-7-fold4.pkl"), 'rb')))
        else:
            cls.model1 = pickle.load(open(os.path.join(model_path, "model_20230102-2-7.pkl"), 'rb'))


        # 潮位ラベル=0
        if cls.emsemble:
            cls.L_models0 = []
            # 使用するモデルを順に読み込む
            cls.L_models0.append(pickle.load(open(os.path.join(model_path, "model_20230102-2-159.pkl"), 'rb')))
            cls.L_models0.append(pickle.load(open(os.path.join(model_path, "model_20230102-2-159-fold0.pkl"), 'rb')))
            cls.L_models0.append(pickle.load(open(os.path.join(model_path, "model_20230102-2-159-fold1.pkl"), 'rb')))
            cls.L_models0.append(pickle.load(open(os.path.join(model_path, "model_20230102-2-159-fold2.pkl"), 'rb')))
            cls.L_models0.append(pickle.load(open(os.path.join(model_path, "model_20230102-2-159-fold3.pkl"), 'rb')))
            cls.L_models0.append(pickle.load(open(os.path.join(model_path, "model_20230102-2-159-fold4.pkl"), 'rb')))
        else:
            cls.model0 = pickle.load(open(os.path.join(model_path, "model_20230102-2-159.pkl"), 'rb'))


        # LabelEncoder用（166箇所の観測所をカテゴリ化）
        cls.le = LabelEncoder()
        cls.le.classes_ = np.load(os.path.join(model_path, "classes-station-166-v2.npy"), allow_pickle=True)


        # 潮位ラベル＋各水位観測所に対応する雨量観測所
        cls.label = pd.read_excel(os.path.join(model_path, "雨量観測所対応and潮位ラベル.xlsx"))

        #　それぞれ対象の水位観測所
        cls.L_liketide = list(cls.label[cls.label["潮位ラベル"]==1]["水位観測所"].unique())
        cls.L_notliketide = list(cls.label[cls.label["潮位ラベル"]==0]["水位観測所"].unique())

        if cls.userain:
            # 雨量観測所は一つにまとめる
            cls.L_rain = list(cls.label["使用する雨量観測所"].unique())
            cls.label = cls.label.drop("潮位ラベル", axis=1)






        # n日前のデータ、初期値はstation以外全てNULL
        # 水位(潮位ラベル=0)
        # 過去データでカラム名が変わるもの
        cls.rename_column1_water_0 = ['mean', 'max', 'min', 'std', 'nullcount', '0-6mean', '6-12mean', '12-18mean', '18-24mean', '0-12mean', '12-24mean']
        cls.rename_column2_water_0 = ['mean', 'max', 'min', 'std', 'nullcount', '0-12mean', '12-24mean']
        # stationだけ入れて他は空のデータフレーム作成
        cls.day1ago_water_0 = pd.DataFrame(columns=['station']+cls.rename_column1_water_0)
        cls.day1ago_water_0["station"] = cls.L_notliketide
        cls.day1ago_water_0[cls.rename_column1_water_0] = np.nan
        cls.day2ago_water_0 = pd.DataFrame(columns=['station']+cls.rename_column2_water_0)
        cls.day2ago_water_0["station"] = cls.L_notliketide
        cls.day2ago_water_0[cls.rename_column2_water_0] = np.nan

        # 水位(潮位ラベル=1)
        # 過去データでカラム名が変わるもの
        cls.rename_column1_water_1 = ['h-0', 'h-1', 'h-2', 'h-3', 'h-4', 'h-5', 'h-6', 'h-7', 'h-8', 'h-9', 'h-10', 'h-11', 'h-12', 'h-13', 'h-14', 'h-15', 'h-16', 'h-17', 'h-18', 'h-19', 'h-20', 'h-21', 'h-22', 'h-23', 'mean', 'max', 'min', 'std', 'nullcount', '0-12mean', '12-24mean']
        cls.rename_column2_water_1 = ['mean', 'max', 'min', 'std', 'nullcount', '0-12mean', '12-24mean']
        cls.rename_column3_water_1 = ['mean', 'std','nullcount']
        cls.rename_column4_water_1 = ['std', 'nullcount']
        # stationだけ入れて他は空のデータフレーム作成
        cls.day1ago_water_1 = pd.DataFrame(columns=['station']+cls.rename_column1_water_1)
        cls.day1ago_water_1["station"] = cls.L_liketide
        cls.day1ago_water_1[cls.rename_column1_water_1] = np.nan
        cls.day2ago_water_1 = cls.day1ago_water_1[['station']+cls.rename_column2_water_1].copy()
        cls.day3ago_water_1 = cls.day2ago_water_1.copy()
        cls.day4ago_water_1 = cls.day3ago_water_1[['station']+cls.rename_column3_water_1].copy()
        cls.day5ago_water_1 = cls.day4ago_water_1.copy()
        cls.day6ago_water_1 = cls.day5ago_water_1[['station']+cls.rename_column4_water_1].copy()
        cls.day7ago_water_1 = cls.day6ago_water_1.copy()
        cls.day8ago_water_1 = cls.day6ago_water_1.copy()
        cls.day9ago_water_1 = cls.day6ago_water_1.copy()
        cls.day10ago_water_1 = cls.day6ago_water_1.copy()
        cls.day11ago_water_1 = cls.day6ago_water_1.copy()
        cls.day12ago_water_1 = cls.day6ago_water_1.copy()
        cls.day13ago_water_1 = cls.day6ago_water_1.copy()
        cls.day14ago_water_1 = cls.day6ago_water_1.copy()
        cls.day15ago_water_1 = cls.day6ago_water_1.copy()
        cls.day16ago_water_1 = cls.day6ago_water_1.copy()


        # 雨量
        if cls.userain:
            # 過去データでカラム名が変わるもの
            cls.rename_column_rain = ['mean_rain']
            # station_rainだけ入れて他は空のデータフレーム作成
            cls.day1ago_rain = pd.DataFrame(columns=['station_rain']
                                                    +cls.rename_column_rain)
            cls.day1ago_rain["station_rain"] = cls.L_rain
            cls.day1ago_rain[cls.rename_column_rain] = np.nan
            cls.day2ago_rain = cls.day1ago_rain.copy()

        # カラム変更用
        cls.d_rename = {}
        # 水位(潮位ラベル=0)
        L0 = []
        for i in range(1,3):
            if i<=1:
                L0.append({x: x+f"_{i}d_ago" for x in cls.rename_column1_water_0})
            else:
                L0.append({x: x+f"_{i}d_ago" for x in cls.rename_column2_water_0})
        else:
            cls.d_rename["water-0"] = L0
        # 水位(潮位ラベル=1)
        L1 = []
        for i in range(1,17):
            if i<=1:
                L1.append({x: x+f"_{i}d_ago" for x in cls.rename_column1_water_1})
            elif i<=3:
                L1.append({x: x+f"_{i}d_ago" for x in cls.rename_column2_water_1})
            elif i<=5:
                L1.append({x: x+f"_{i}d_ago" for x in cls.rename_column3_water_1})
            else:
                L1.append({x: x+f"_{i}d_ago" for x in cls.rename_column4_water_1})
        else:
            cls.d_rename["water-1"] = L1
        #雨量
        if cls.userain:
            L2 = []
            for i in range(1,3):
                L2.append({"mean_rain": f"mean_{i}d_ago_rain"})
            else:
                cls.d_rename["rain"] = L2

        cls.L_h = []
        for i in range(24):
            cls.L_h.append(f'h-{i}')


        # 想定の特徴量
        cls.feature_col_0 = ['hour_sin', 'hour_cos', 'station_c', 'h-0', 'h-6', 'h-12', 'h-15', 'h-18', 'h-21', 'h-22', 'h-23', 'mean', 'max', 'min', 'std', 'nullcount', '0-12mean', '12-24mean', '0-6mean', '6-12mean', '12-18mean', '18-24mean', 'std_1d_ago', 'nullcount_1d_ago', '0-6mean_1d_ago', '6-12mean_1d_ago', '12-18mean_1d_ago', '18-24mean_1d_ago', 'std_2d_ago', 'nullcount_2d_ago', '0-12mean_2d_ago', '12-24mean_2d_ago', '12-14-mean_rain', '12-14-max_rain', '15-17-mean_rain', '15-17-max_rain', '18-20-mean_rain', '18-20-max_rain', '21-23-mean_rain', '21-23-max_rain', 'mean_rain', 'mean_1d_ago_rain', 'mean_2d_ago_rain']

        cls.feature_col_１ = ['hour_sin', 'hour_cos', 'station_c', 'h-0', 'h-1', 'h-2', 'h-3', 'h-4', 'h-5', 'h-6', 'h-7', 'h-8', 'h-9', 'h-10', 'h-11', 'h-12', 'h-13', 'h-14', 'h-15', 'h-16', 'h-17', 'h-18', 'h-19', 'h-20', 'h-21', 'h-22', 'h-23', 'mean', 'max', 'min', 'std', 'nullcount', '0-12mean', '12-24mean', 'h-0_1d_ago', 'h-1_1d_ago', 'h-2_1d_ago', 'h-3_1d_ago', 'h-4_1d_ago', 'h-5_1d_ago', 'h-6_1d_ago', 'h-7_1d_ago', 'h-8_1d_ago', 'h-9_1d_ago', 'h-10_1d_ago', 'h-11_1d_ago', 'h-12_1d_ago', 'h-13_1d_ago', 'h-14_1d_ago', 'h-15_1d_ago', 'h-16_1d_ago', 'h-17_1d_ago', 'h-18_1d_ago', 'h-19_1d_ago', 'h-20_1d_ago', 'h-21_1d_ago', 'h-22_1d_ago', 'h-23_1d_ago', 'mean_1d_ago', 'max_1d_ago', 'min_1d_ago', 'std_1d_ago', 'nullcount_1d_ago', '0-12mean_1d_ago', '12-24mean_1d_ago', 'mean_2d_ago', 'max_2d_ago', 'min_2d_ago', 'std_2d_ago', 'nullcount_2d_ago', '0-12mean_2d_ago', '12-24mean_2d_ago', 'mean_3d_ago', 'max_3d_ago', 'min_3d_ago', 'std_3d_ago', 'nullcount_3d_ago', '0-12mean_3d_ago', '12-24mean_3d_ago', 'mean_4d_ago', 'nullcount_4d_ago', 'mean_5d_ago', 'nullcount_5d_ago', 'std_14d_ago', 'nullcount_14d_ago', 'std_15d_ago', 'nullcount_15d_ago', 'std_16d_ago', 'nullcount_16d_ago', '12-14-mean_rain', '12-14-max_rain', '15-17-mean_rain', '15-17-max_rain', '18-20-mean_rain', '18-20-max_rain', '21-23-mean_rain', '21-23-max_rain', 'mean_rain', 'mean_1d_ago_rain', 'mean_2d_ago_rain']


        return True


    @classmethod
    def predict(cls, input):
        """Predict method

        Args:
            input: Data of the sample you want to make inference from (dict)

        Returns:
            list: Inference for the given input.

        """
        # 評価対象の水位観測所
        stations = input['stations']
        # 取得出来た水位データ
        waterlevel = input['waterlevel']

        # 評価対象のものに絞る(無い日付のデータはどうする？)
        merged_water1 = pd.merge(pd.DataFrame(cls.L_liketide, columns=['station']), pd.DataFrame(waterlevel))
        merged_water0 = pd.merge(pd.DataFrame(cls.L_notliketide, columns=['station']), pd.DataFrame(waterlevel))



        # ソート(不要か)
        merged_water1 = merged_water1.sort_values(["station", "hour"], ascending=True)
        merged_water0 = merged_water0.sort_values(["station", "hour"], ascending=True)


        # 欠損はNaNに
        merged_water1['value'] = pd.to_numeric(merged_water1['value'], errors='coerce')
        merged_water0['value'] = pd.to_numeric(merged_water0['value'], errors='coerce')

        # 雨量
        if cls.userain:
            rainfall = input['rainfall']
            merged_rain = pd.merge(pd.DataFrame(cls.L_rain, columns=['station']), pd.DataFrame(rainfall)).rename(columns={"station":"station_rain"})
            merged_rain = merged_rain.sort_values(["station_rain", "hour"], ascending=True)
            merged_rain['value'] = pd.to_numeric(merged_rain['value'], errors='coerce')


        # 特徴量作成
        # 雨量
        if cls.userain:
            merged_rain = merged_rain.drop("city", axis=1)
            merged_rain["hour"] = "h-" + merged_rain['hour'].astype(str)
            merged_rain2 = merged_rain.pivot_table(index=['station_rain'], columns=['hour'], values='value').reset_index()
            # 存在しない時刻の場合は空のデータフレームを
            for x in cls.L_h:
                if x not in list(merged_rain2.columns):
                    merged_rain2[x]=np.nan
            # 順番を揃える
            merged_rain2_cp = merged_rain2.copy()[['station_rain']+cls.L_h]
            # 3時間おきで
            merged_rain2['12-14-mean_rain'] = merged_rain2_cp[merged_rain2_cp.columns[13:16]].mean(axis=1)
            merged_rain2['12-14-max_rain'] = merged_rain2_cp[merged_rain2_cp.columns[13:16]].max(axis=1)
            merged_rain2['15-17-mean_rain'] = merged_rain2_cp[merged_rain2_cp.columns[16:19]].mean(axis=1)
            merged_rain2['15-17-max_rain'] = merged_rain2_cp[merged_rain2_cp.columns[16:19]].max(axis=1)
            merged_rain2['18-20-mean_rain'] = merged_rain2_cp[merged_rain2_cp.columns[19:22]].mean(axis=1)
            merged_rain2['18-20-max_rain'] = merged_rain2_cp[merged_rain2_cp.columns[19:22]].max(axis=1)
            merged_rain2['21-23-mean_rain'] = merged_rain2_cp[merged_rain2_cp.columns[22:25]].mean(axis=1)
            merged_rain2['21-23-max_rain'] = merged_rain2_cp[merged_rain2_cp.columns[22:25]].max(axis=1)
            # 1日の平均も追加
            merged_rain2["mean_rain"] = merged_rain2_cp[merged_rain2_cp.columns[1:]].mean(axis=1)

            cls.today_rain = merged_rain2.loc[:,['station_rain', 'mean_rain','12-14-mean_rain', '12-14-max_rain', '15-17-mean_rain', '15-17-max_rain', '18-20-mean_rain', '18-20-max_rain', '21-23-mean_rain', '21-23-max_rain']].copy()

        # 水位(潮位ラベル=0)
        df_group1_water0 = merged_water0.groupby("station").value.agg(["mean", "min", "max", "std"]).reset_index()
        merged_water0_ = merged_water0.drop("river",axis=1).copy()
        merged_water0_["hour"] = "h-" + merged_water0_['hour'].astype(str)
        df_group2_water0 = merged_water0_.pivot_table(index=['station'], columns=['hour'], values='value').reset_index()
        # 存在しない時刻の場合は空のデータフレームを
        for x in cls.L_h:
            if x not in list(df_group2_water0.columns):
                df_group2_water0[x]=np.nan
        # 順番を揃える
        df_group2_water0_cp = df_group2_water0.copy()[['station']+cls.L_h]
        # その他特徴量作成
        df_group2_water0["nullcount"] = df_group2_water0_cp[df_group2_water0_cp.columns[1:]].count(numeric_only=True,axis=1)
        df_group2_water0["0-12mean"] = df_group2_water0_cp[df_group2_water0_cp.columns[1:13]].mean(axis=1)
        df_group2_water0["12-24mean"] = df_group2_water0_cp[df_group2_water0_cp.columns[13:25]].mean(axis=1)
        df_group2_water0["0-6mean"] = df_group2_water0_cp[df_group2_water0_cp.columns[1:7]].mean(axis=1)
        df_group2_water0["6-12mean"] = df_group2_water0_cp[df_group2_water0_cp.columns[7:13]].mean(axis=1)
        df_group2_water0["12-18mean"] = df_group2_water0_cp[df_group2_water0_cp.columns[13:19]].mean(axis=1)
        df_group2_water0["18-24mean"] = df_group2_water0_cp[df_group2_water0_cp.columns[19:25]].mean(axis=1)

        cls.today_water_0 = pd.merge(df_group1_water0, df_group2_water0, on="station")
        cls.today_water_0 = cls.today_water_0[['station']+['h-0', 'h-6', 'h-12', 'h-15', 'h-18', 'h-21', 'h-22', 'h-23', 'mean', 'max', 'min', 'std', 'nullcount', '0-12mean', '12-24mean', '0-6mean', '6-12mean', '12-18mean', '18-24mean']]
        merged_water0 = pd.merge(merged_water0, cls.today_water_0, on="station", how="left")
        merged_water0 = pd.merge(merged_water0, cls.day1ago_water_0.rename(columns=cls.d_rename["water-0"][0]), on="station", how="left")
        merged_water0 = pd.merge(merged_water0, cls.day2ago_water_0.rename(columns=cls.d_rename["water-0"][1]), on="station", how="left")

        # 雨量結合
        if cls.userain:
            merged_water0 = pd.merge(merged_water0, cls.label, left_on="station", right_on="水位観測所", how="left")
            merged_water0 = merged_water0.drop('水位観測所',axis=1)
            merged_water0 = pd.merge(merged_water0, cls.today_rain, left_on="使用する雨量観測所", right_on="station_rain", how="left")
            merged_water0 = merged_water0.drop('station_rain',axis=1)
            merged_water0 = pd.merge(merged_water0, cls.day1ago_rain.rename(columns=cls.d_rename["rain"][0]),
                                     left_on="使用する雨量観測所", right_on="station_rain", how="left")
            merged_water0 = merged_water0.drop('station_rain',axis=1)
            merged_water0 = pd.merge(merged_water0, cls.day2ago_rain.rename(columns=cls.d_rename["rain"][1]),
                                     left_on="使用する雨量観測所", right_on="station_rain", how="left")
            merged_water0 = merged_water0.drop(['station_rain', '使用する雨量観測所'],axis=1)

        # 水位(潮位ラベル=1)
        df_group1_water1 = merged_water1.groupby("station").value.agg(["mean", "min", "max", "std"]).reset_index()
        merged_water1_ = merged_water1.drop("river",axis=1).copy()
        merged_water1_["hour"] = "h-" + merged_water1_['hour'].astype(str)
        df_group2_water1 = merged_water1_.pivot_table(index=['station'], columns=['hour'], values='value').reset_index()
        # 存在しない時刻の場合は空のデータフレームを
        for x in cls.L_h:
            if x not in list(df_group2_water1.columns):
                df_group2_water1[x]=np.nan
        # 順番を揃える
        df_group2_water1_cp = df_group2_water1.copy()[['station']+cls.L_h]
        # その他特徴量作成
        df_group2_water1["nullcount"] = df_group2_water1_cp[df_group2_water1_cp.columns[1:]].count(numeric_only=True,axis=1)
        df_group2_water1["0-12mean"] = df_group2_water1_cp[df_group2_water1_cp.columns[1:13]].mean(axis=1)
        df_group2_water1["12-24mean"] = df_group2_water1_cp[df_group2_water1_cp.columns[13:25]].mean(axis=1)


        cls.today_water_1 = pd.merge(df_group1_water1, df_group2_water1, on="station")

        cls.today_water_1 = cls.today_water_1[['station']+cls.rename_column1_water_1]
        merged_water1 = pd.merge(merged_water1, cls.today_water_1, on="station", how="left")
        merged_water1 = pd.merge(merged_water1, cls.day1ago_water_1.rename(columns=cls.d_rename["water-1"][0]), on="station", how="left")
        merged_water1 = pd.merge(merged_water1, cls.day2ago_water_1.rename(columns=cls.d_rename["water-1"][1]), on="station", how="left")
        merged_water1 = pd.merge(merged_water1, cls.day3ago_water_1.rename(columns=cls.d_rename["water-1"][2]), on="station", how="left")
        merged_water1 = pd.merge(merged_water1, cls.day4ago_water_1.rename(columns=cls.d_rename["water-1"][3]), on="station", how="left")
        merged_water1 = pd.merge(merged_water1, cls.day5ago_water_1.rename(columns=cls.d_rename["water-1"][4]), on="station", how="left")
        merged_water1 = pd.merge(merged_water1, cls.day14ago_water_1.rename(columns=cls.d_rename["water-1"][13]), on="station", how="left")
        merged_water1 = pd.merge(merged_water1, cls.day15ago_water_1.rename(columns=cls.d_rename["water-1"][14]), on="station", how="left")
        merged_water1 = pd.merge(merged_water1, cls.day16ago_water_1.rename(columns=cls.d_rename["water-1"][15]), on="station", how="left")

        # 雨量結合
        if cls.userain:
            merged_water1 = pd.merge(merged_water1, cls.label, left_on="station", right_on="水位観測所", how="left")
            merged_water1 = merged_water1.drop('水位観測所',axis=1)
            merged_water1 = pd.merge(merged_water1, cls.today_rain, left_on="使用する雨量観測所", right_on="station_rain", how="left")
            merged_water1 = merged_water1.drop('station_rain',axis=1)
            merged_water1 = pd.merge(merged_water1, cls.day1ago_rain.rename(columns=cls.d_rename["rain"][0]), left_on="使用する雨量観測所", right_on="station_rain", how="left")
            merged_water1 = merged_water1.drop('station_rain',axis=1)
            merged_water1 = pd.merge(merged_water1, cls.day2ago_rain.rename(columns=cls.d_rename["rain"][1]),
                                     left_on="使用する雨量観測所", right_on="station_rain", how="left")
            merged_water1 = merged_water1.drop('station_rain',axis=1)


        # 時間変数を変換
        # 水位(潮位ラベル=0)
        merged_water0["hour_sin"] = np.sin(np.pi*merged_water0["hour"]/12)
        merged_water0["hour_cos"] = np.cos(np.pi*merged_water0["hour"]/12)
        # labelencode
        merged_water0["station_c"] = cls.le.transform(merged_water0["station"])
        # 水位(潮位ラベル=1)
        merged_water1["hour_sin"] = np.sin(np.pi*merged_water1["hour"]/12)
        merged_water1["hour_cos"] = np.cos(np.pi*merged_water1["hour"]/12)
        # labelencode
        merged_water1["station_c"] = cls.le.transform(merged_water1["station"])

        # 推論
        # モデル一つだけ
        if not cls.emsemble:
            # 水位(潮位ラベル=0)
            merged_water0["value"] = cls.model0.predict(merged_water0[cls.feature_col_0], num_iteration=cls.model0.best_iteration)
            # 水位(潮位ラベル=1)
            merged_water1["value"] = cls.model1.predict(merged_water1[cls.feature_col_1], num_iteration=cls.model1.best_iteration)
        # 4,5年分モデル2つ
        else:
            # 水位(潮位ラベル=0)
            merged_water0["value"] = 0
            for model in cls.L_models0:
                merged_water0["value"] += model.predict(merged_water0[cls.feature_col_0], num_iteration=model.best_iteration)/len(cls.L_models0)

            # 水位(潮位ラベル=1)
            merged_water1["value"] = 0
            for model in cls.L_models1:
                merged_water1["value"] += model.predict(merged_water1[cls.feature_col_1], num_iteration=model.best_iteration)/len(cls.L_models1)


        # 予測結果
        prediction = pd.concat([merged_water0[['hour', 'station', 'value']], merged_water1[['hour', 'station', 'value']]], axis=0).to_dict('records')
        # prediction = merged_water0[merged_water0["station"]=="白川"][['hour', 'station', 'value']].to_dict('records')

        #翌日の推論に向けた準備
        # 雨量
        if cls.userain:
            cls.day2ago_rain = cls.day1ago_rain
            cls.day1ago_rain = cls.today_rain[['station_rain', 'mean_rain']]

        # 水位(潮位ラベル=0)
        cls.day2ago_water_0 = cls.day1ago_water_0[['station']+cls.rename_column2_water_0]
        cls.day1ago_water_0 = cls.today_water_0[['station']+cls.rename_column1_water_0]

        # 水位(潮位ラベル=1)
        cls.day16ago_water_1 = cls.day15ago_water_1
        cls.day15ago_water_1 = cls.day14ago_water_1
        cls.day14ago_water_1 = cls.day13ago_water_1
        cls.day13ago_water_1 = cls.day12ago_water_1
        cls.day12ago_water_1 = cls.day11ago_water_1
        cls.day11ago_water_1 = cls.day10ago_water_1
        cls.day10ago_water_1 = cls.day9ago_water_1
        cls.day9ago_water_1 = cls.day8ago_water_1
        cls.day8ago_water_1 = cls.day7ago_water_1
        cls.day7ago_water_1 = cls.day6ago_water_1
        cls.day6ago_water_1 = cls.day5ago_water_1[['station']+cls.rename_column4_water_1]
        cls.day5ago_water_1 = cls.day4ago_water_1
        cls.day4ago_water_1 = cls.day3ago_water_1[['station']+cls.rename_column3_water_1]
        cls.day3ago_water_1 = cls.day2ago_water_1
        cls.day2ago_water_1 = cls.day1ago_water_1[['station']+cls.rename_column2_water_1]
        cls.day1ago_water_1 = cls.today_water_1[['station']+cls.rename_column1_water_1]

        return prediction
