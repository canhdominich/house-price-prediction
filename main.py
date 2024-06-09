import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc # Garbage Collector

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    return pd.read_csv(file_path, low_memory=False)

def load_properties(file_path):
    try:
        property_data_df = load_data(file_path)
        print("property_data_df:", property_data_df.shape)

        print(property_data_df.head())

        cat_vars = [
            'airconditioningtypeid', 'architecturalstyletypeid', 'buildingqualitytypeid', 'buildingclasstypeid',
            'decktypeid', 'fips', 'hashottuborspa', 'fireplaceflag', 'heatingorsystemtypeid',
            'propertycountylandusecode', 'propertylandusetypeid', 'propertyzoningdesc', 'regionidcity',
            'regionidcounty', 'regionidneighborhood', 'regionidzip', 'storytypeid', 'typeconstructiontypeid',
            'yearbuilt', 'taxdelinquencyflag', 'latitude', 'longitude', 'parcelid', 'assessmentyear'
        ]

        num_vars = [i for i in property_data_df.columns if i not in cat_vars]
        print ("Có {} numeric và {} categorical columns".format(len(num_vars),len(cat_vars)))

        # Correlation matrix plot
        corr = property_data_df[num_vars].corr()

        # Tạo một ma trận mask có cùng kích thước với ma trận tương quan corr, với tất cả các giá trị ban đầu là False. 
        # np.bool là kiểu dữ liệu boolean trong NumPy
        # Tuy nhiên, trong các phiên bản mới hơn của NumPy, sẽ sử dụng np.bool_ thay vì np.bool
        mask = np.zeros_like(corr, dtype=np.bool_)

        mask[np.triu_indices_from(mask)] = True
        # print("mask=", mask, len(mask))

        # Thiết lập một đối tượng Figure và Axes của Matplotlib với kích thước 19x19 inch.
        f, ax = plt.subplots(figsize=(19, 19))

        # Draw the heatmap with the mask and correct aspect ratio
        # mask=mask: Chỉ hiển thị tam giác dưới của ma trận tương quan (bởi vì tam giác trên đã bị mặt nạ che khuất).
        # cmap='coolwarm': Sử dụng bảng màu 'coolwarm'.
        # vmax=1: Thiết lập giá trị tối đa của thang màu là 1.
        # center=0: Trung tâm của bảng màu là 0.
        # annot=True: Hiển thị giá trị tương quan trong các ô của heatmap.
        # square=True: Mỗi ô của heatmap sẽ có hình vuông.
        # linewidths=.3: Thiết lập độ rộng của đường viền giữa các ô là 0.3.
        # cbar_kws={"shrink": .5}: Thu nhỏ thanh màu (color bar) bằng 50%.
        ax = sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=1, center=0, annot=True,
                    square=True, linewidths=.3, cbar_kws={"shrink": .5})
        
        # Xoay nhãn trục x 45 độ và căn chỉnh sang phải để dễ đọc hơn.
        ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right')

        # Lưu lại ảnh heatmap
        # dpi=300: Thiết lập độ phân giải của ảnh là 300 DPI (dots per inch), đảm bảo ảnh có chất lượng cao.
        # bbox_inches='tight': Đảm bảo rằng các nhãn trục không bị cắt khỏi ảnh lưu.
        plt.savefig('images/heatmap.png', dpi=300, bbox_inches='tight')
        # plt.show()
        
        del corr, ax
        gc.collect()
        print('Memory usage reduction…')

    except Exception as e:
        print("Errors: {}".format(e))

def show_missing_value(file_path):
    try:
        property_data_df = load_data(file_path)
        # Tính toán tổng số giá trị thiếu (NaN) trong từng cột của DataFrame property_data_df. 
        # Kết quả được trả về dưới dạng Series, sau đó được đặt lại chỉ mục (reset_index) để chuyển thành DataFrame missing_df.
        missing_df = property_data_df.isnull().sum(axis=0).reset_index()

        # Đặt lại tên các cột trong missing_df thành column_name và missing_count, giúp dữ liệu dễ đọc và hiểu hơn.
        missing_df.columns = ['column_name', 'missing_count']

        # Lọc DataFrame missing_df để chỉ giữ lại các cột có số lượng giá trị thiếu lớn hơn 0.
        missing_df = missing_df.loc[missing_df['missing_count']>0]

        # Sắp xếp DataFrame missing_df theo thứ tự tăng dần của số lượng giá trị thiếu.
        missing_df = missing_df.sort_values(by='missing_count')

        ind = np.arange(missing_df.shape[0])
        fig, ax = plt.subplots(figsize=(12,18))

        # Vẽ biểu đồ cột ngang (barh) với các chỉ số ind và giá trị số lượng giá trị thiếu từ missing_df, màu của các cột là màu xanh dương.
        ax.barh(ind, missing_df.missing_count.values, color='blue')

        ax.set_yticks(ind)
        ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
        ax.set_xlabel("Count of missing values")
        ax.set_title("Number of missing values in each column")

        plt.savefig('images/missing_values.png', dpi=300, bbox_inches='tight')
        # plt.show()

        del ax
        gc.collect()
        print('Memory usage reduction…')

        return property_data_df

    except Exception as e:
        print("Errors: {}".format(e))

def analyse_missing_value(file_path):
    df = load_data(file_path)

    missing_df = df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df['missing_ratio'] = missing_df['missing_count'] / df.shape[0]
    missing_ratio_gt_0_99 = missing_df.loc[missing_df['missing_ratio']>0.99]

    # miss > 99%
    print("Missing values have a ratio greater than 99%", missing_ratio_gt_0_99)
    # Trước khi quyết định loại bỏ một số feature (hầu hết bị thiếu),cần đảm bảo rằng một ô có chữ "NaN" thực sự mang ý nghĩa là "Không".

# Xử lý các dữ liệu thiếu liên quan đến bể bơi và bồn tắm nóng
def handle_poolhottubor(property_data):
    # Inplace: Đối số này cho phép việc thay thế được thực hiện trực tiếp trên DataFrame gốc mà không cần tạo ra một bản sao mới
    
    # 0 pools
    property_data.poolcnt.fillna(0,inplace = True)
    # 0 hot tubs or spas
    property_data.hashottuborspa.fillna(0,inplace = True)
    # Convert "True" to 1
    property_data.hashottuborspa.replace(to_replace = True, value = 1,inplace = True) # Category

    # Set properties that have a pool but no info on poolsize equal to the median (Trung vị) poolsize value.
    property_data.loc[property_data.poolcnt==1, 'poolsizesum'] = property_data.loc[property_data.poolcnt==1, 'poolsizesum'].fillna(property_data[property_data.poolcnt==1].poolsizesum.median())
    # "0 pools" = "0 sq ft of pools"
    property_data.loc[property_data.poolcnt==0, 'poolsizesum']=0

    # "0 pools with a spa/hot tub"
    property_data.pooltypeid2.fillna(0,inplace = True)
    # "0 pools without a hot tub"
    property_data.pooltypeid7.fillna(0,inplace = True)

    # Drop redundant feature
    property_data.drop('pooltypeid10', axis=1, inplace=True)
    
    return property_data

def handle_fireplace(property_data):
    # If fireplaceflag = True and fireplacecnt = NaN then set fireplacecnt = 1
    property_data.loc[(property_data['fireplaceflag'] == True) & (property_data['fireplacecnt'].isnull()), ['fireplacecnt']] = 1
    # If "fireplacecnt" is 1 or larger "fireplaceflag" is "NaN" then set "fireplaceflag" to "True".
    property_data.loc[(property_data['fireplacecnt'] >= 1.0) & (property_data['fireplaceflag'].isnull()), ['fireplaceflag']] = True # Category
    
    # Convert "NaN" to 0
    property_data.fireplaceflag.fillna(0,inplace = True)
    # Convert "True" to 1
    property_data.fireplaceflag.replace(to_replace = True, value = 1,inplace = True)
    
    # If 'fireplacecnt' is "NaN", replace with "0"
    property_data.fireplacecnt.fillna(0,inplace = True)
    
    return property_data

def handle_garage(property_data):
    # Total number of garages on the lot including an attached garage, if 'NaN' then replace by 0
    property_data.garagecarcnt.fillna(0,inplace = True)
    # Total number of square feet of all garages on lot including an attached garage
    property_data.garagetotalsqft.fillna(0,inplace = True)
    return property_data

def handle_tax(property_data):
    # Replace "NaN" with "0"
    property_data.taxdelinquencyflag.fillna(0,inplace = True)
    # Change "Y" to "1"
    property_data.taxdelinquencyflag.replace(to_replace = 'Y', value = 1,inplace = True) # Category
    
    property_data.landtaxvaluedollarcnt.fillna(0,inplace = True)
    property_data.structuretaxvaluedollarcnt.fillna(0,inplace = True)

    property_data['taxvaluedollarcnt'].fillna((property_data['taxvaluedollarcnt'].mean()), inplace=True)
    
    # Drop "regionidcity"
    property_data.drop('regionidcity', axis=1, inplace=True) # Category

    # Fill in "NaN" "yearbuilt" with most common
    yearbuilt = property_data['yearbuilt'].value_counts().idxmax()
    property_data['yearbuilt'] = property_data['yearbuilt'].fillna(yearbuilt) # Category
    
    return property_data

def handle_squarefeet(property_data):
    # Drop "finishedsquarefeet6"
    property_data.drop('finishedsquarefeet6', axis=1, inplace=True)
    # Drop "finishedsquarefeet12"
    property_data.drop('finishedsquarefeet12', axis=1, inplace=True)
    # Drop "finishedfloor1squarefeet"
    property_data.drop('finishedfloor1squarefeet', axis=1, inplace=True)

    # Replace "NaN" "calculatedfinishedsquarefeet" values with mean.
    property_data['calculatedfinishedsquarefeet'].fillna((property_data['calculatedfinishedsquarefeet'].mean()), inplace=True)

    # If "numberofstories" is equal to "1", then we can replace the "NaN"s with the "calculatedfinishedsquarefeet" value. Fill in the rest with the average values.
    property_data.loc[property_data['numberofstories'] == 1.0,'finishedsquarefeet50'] = property_data['calculatedfinishedsquarefeet']
    property_data['finishedsquarefeet50'].fillna((property_data['finishedsquarefeet50'].mean()), inplace=True)

    # Replace "NaN" "finishedsquarefeet15" values with calculatedfinishedsquarefeet.
    property_data.loc[property_data['finishedsquarefeet15'].isnull(),'finishedsquarefeet15'] = property_data['calculatedfinishedsquarefeet']
    # Replace rest valule "NaN" "finishedsquarefeet15" values with mean.
    property_data['finishedsquarefeet15'].fillna((property_data['finishedsquarefeet15'].mean()), inplace=True)
    # Change numberofstories with common value 
    property_data['numberofstories'].fillna(1,inplace = True)
    
    return property_data

def handle_bathroom(property_data):
    # Drop "threequarterbathnbr"
    property_data.drop('threequarterbathnbr', axis=1, inplace=True)
    # Drop "fullbathcnt"
    property_data.drop('fullbathcnt', axis=1, inplace=True)

    # Fill in "NaN" "calculatedbathnbr" with most common
    bathroommode = property_data['calculatedbathnbr'].value_counts().idxmax()
    property_data['calculatedbathnbr'] = property_data['calculatedbathnbr'].fillna(bathroommode)
    return property_data

def handle_data_rest(property_data):
    # Drop "taxdelinquencyyear"
    property_data.drop('taxdelinquencyyear', axis=1, inplace=True)
    # Drop 'basementsqft'
    property_data.drop('basementsqft', axis=1, inplace = True)
    # Drop "storytypeid"
    property_data.drop('storytypeid', axis=1, inplace=True) # Category
    # Drop "architecturalstyletypeid"
    property_data.drop('architecturalstyletypeid', axis=1, inplace=True) # Category
    # Drop "typeconstructiontypeid" and "finishedsquarefeet13"
    property_data.drop('typeconstructiontypeid', axis=1, inplace=True) # Category
    property_data.drop('finishedsquarefeet13', axis=1, inplace=True)
    # Drop "buildingclasstypeid"
    property_data.drop('buildingclasstypeid', axis=1, inplace=True) # Category

    # Replace 'yardbuildingsqft17' "NaN"s with "0".
    property_data.yardbuildingsqft17.fillna(0,inplace = True)
    # Replace 'yardbuildingsqft26' "NaN"s with "0".
    property_data.yardbuildingsqft26.fillna(0,inplace = True)
    # Change "decktypeid" "Nan"s to "0"
    property_data.decktypeid.fillna(0,inplace = True)
    # Convert "decktypeid" "66.0" to "1"
    property_data.decktypeid.replace(to_replace = 66.0, value = 1,inplace = True) # Category
    # Change "airconditioningtypeid" NaN to "5"
    property_data.airconditioningtypeid.fillna(5,inplace = True) # Category
    # Change "heatingorsystemtypeid" NaN to "13"
    property_data.heatingorsystemtypeid.fillna(13,inplace = True) # Category

    # Fill in "NaN" "buildingqualitytypeid" bằng giá trị phổ biến
    buildingqual = property_data['buildingqualitytypeid'].value_counts().idxmax()
    property_data['buildingqualitytypeid'] = property_data['buildingqualitytypeid'].fillna(buildingqual) # Category
    # Fill in "NaN" "unitcnt" bằng giá trị phổ biến
    unitcommon = property_data['unitcnt'].value_counts().idxmax()
    property_data['unitcnt'] = property_data['unitcnt'].fillna(unitcommon)
    

    property_data['lotsizesquarefeet'].fillna((property_data['lotsizesquarefeet'].mean()), inplace=True)

    # Drop "regionidneighborhood"
    property_data.drop('regionidneighborhood', axis=1, inplace=True) # Category
    # Drop 'regionidcounty'
    property_data.drop('regionidcounty', axis=1, inplace=True) # Category
    
    return property_data

def fill_common_value(property_data):    
    # Drop "censustractandblock"
    property_data.drop('censustractandblock', axis=1, inplace=True)

    # Fill in "regionidzip" bằng giá trị phổ biến
    regionidzip = property_data['regionidzip'].value_counts().idxmax()
    property_data['regionidzip'] = property_data['regionidzip'].fillna(regionidzip) # Category

    # Fill in "fips" bằng giá trị phổ biến
    fips = property_data['fips'].value_counts().idxmax()
    property_data['fips'] = property_data['fips'].fillna(fips) # Category

    # Fill in "propertylandusetypeid" bằng giá trị phổ biến
    propertylandusetypeid = property_data['propertylandusetypeid'].value_counts().idxmax()
    property_data['propertylandusetypeid'] = property_data['propertylandusetypeid'].fillna(propertylandusetypeid) # Category

    # Fill in "latitude"  bằng giá trị phổ biến
    latitude = property_data['latitude'].value_counts().idxmax()
    property_data['latitude'] = property_data['latitude'].fillna(latitude) # Category

    # Fill in "longitude" bằng giá trị phổ biến
    longitude = property_data['longitude'].value_counts().idxmax()
    property_data['longitude'] = property_data['longitude'].fillna(longitude) # Category
    
    # Normal value
    property_data[['latitude', 'longitude']] /= 1e6
    property_data['rawcensustractandblock'] /= 1e6

    # Fill in "rawcensustractandblock" bằng giá trị phổ biến
    rawcensustractandblock = property_data['rawcensustractandblock'].value_counts().idxmax()
    property_data['rawcensustractandblock'] = property_data['rawcensustractandblock'].fillna(rawcensustractandblock)

    # Fill in "assessmentyear" bằng giá trị phổ biến
    assessmentyear = property_data['assessmentyear'].value_counts().idxmax()
    property_data['assessmentyear'] = property_data['assessmentyear'].fillna(assessmentyear) # Category

    # Fill in "bedroomcnt" bằng giá trị phổ biến
    bedroomcnt = property_data['bedroomcnt'].value_counts().idxmax()
    property_data['bedroomcnt'] = property_data['bedroomcnt'].fillna(bedroomcnt)

    # Fill in "bathroomcnt" bằng giá trị phổ biến
    bathroomcnt = property_data['bathroomcnt'].value_counts().idxmax()
    property_data['bathroomcnt'] = property_data['bathroomcnt'].fillna(bathroomcnt)

    # Fill in "roomcnt" bằng giá trị phổ biến
    roomcnt = property_data['roomcnt'].value_counts().idxmax()
    property_data['roomcnt'] = property_data['roomcnt'].fillna(roomcnt)
    
    # Fill in "propertycountylandusecode" bằng giá trị phổ biến
    propertycountylandusecode = property_data['propertycountylandusecode'].value_counts().idxmax()
    property_data['propertycountylandusecode'] = property_data['propertycountylandusecode'].fillna(propertycountylandusecode) # Category
    
    # Fill in "NaN" "propertyzoningdesc" with most common
    propertyzoningdesc = property_data['propertyzoningdesc'].value_counts().idxmax()
    property_data['propertyzoningdesc'] = property_data['propertyzoningdesc'].fillna(propertyzoningdesc) # Category
    
    return property_data

def pre_processing_data(property_data):
    property_data = handle_poolhottubor(property_data)
    property_data = handle_fireplace(property_data)
    property_data = handle_garage(property_data)
    property_data = handle_tax(property_data)
    property_data = handle_squarefeet(property_data)
    property_data = handle_bathroom(property_data)
    property_data = handle_data_rest(property_data)
    property_data = fill_common_value(property_data)
    return property_data

def feature_engineering(property_data):
    property_data['avg_garage_size'] = property_data['garagetotalsqft'] / property_data['garagecarcnt']
    property_data['avg_garage_size'].fillna(0, inplace=True)
    
    # Rotated Coordinates
    # property_data['location_1'] = property_data['latitude'] + property_data['longitude']
    # property_data['location_2'] = property_data['latitude'] - property_data['longitude']
    # property_data['location_3'] = property_data['latitude'] + 0.5 * property_data['longitude']
    # property_data['location_4'] = property_data['latitude'] - 0.5 * property_data['longitude']

    property_data['taxpercentage'] = property_data['taxamount'] / property_data['taxvaluedollarcnt']
    property_data['taxpercentage'].fillna((property_data['taxpercentage'].mean()), inplace=True)

    # Drop "taxamount"
    property_data.drop('taxamount', axis=1, inplace=True)

    # Create total_room property equal bathroom_cnt + bedroom_cnt
    property_data['total_room'] = property_data['bedroomcnt'] + property_data['bathroomcnt']
    
    return property_data

def convert_data(property_data):
    # LabelEncoder là một công cụ để chuyển đổi các nhãn văn bản (string labels) thành các số nguyên duy nhất
    countylandusecode = LabelEncoder()
    property_data["propertycountylandusecode"] = countylandusecode.fit_transform(property_data["propertycountylandusecode"])
    zoningdesc = LabelEncoder()
    property_data["propertyzoningdesc"] = zoningdesc.fit_transform(property_data["propertyzoningdesc"])
    return property_data

def transform_data(property_data, property_data_df):
    scaler = StandardScaler()
    # From airconditioningtypeid to the end
    property_data = scaler.fit_transform(property_data.loc[:, 'airconditioningtypeid':])
    property_data = pd.DataFrame(property_data, columns=property_data_df.loc[:, 'airconditioningtypeid':].columns)
    property_data['parcelid'] = property_data_df['parcelid']
    return property_data

def convert_transactiondate(df):
    df["transactiondate_x"] = pd.to_datetime(df["transactiondate"])
    df["year"]              = df["transactiondate_x"].dt.year
    df["month"]             = df["transactiondate_x"].dt.month
    df["day"]               = df["transactiondate_x"].dt.day
    df["weekday"]           = df["transactiondate_x"].dt.weekday
    df["weekofyear"]        = df["transactiondate_x"].dt.isocalendar().week
    df["is_weekend"]        = df["weekday"].apply(lambda x: 1 if x >= 5 else 0)
    df["is_month_end"]      = df["transactiondate_x"].dt.is_month_end.astype(int)
    df["is_month_start"]    = df["transactiondate_x"].dt.is_month_start.astype(int)
    df = df.drop(columns=['transactiondate_x'])
    return df

def evaluate_model(y_tests, y_preds, kf):
    mae, mse, me, r2, ev = 0, 0, 0, 0, 0
    for y_test, y_pred in zip(y_tests, y_preds):
        mae += mean_absolute_error(y_test, y_pred)
        mse += mean_squared_error(y_test, y_pred)
        me  += max_error(y_test, y_pred)
        r2  += r2_score(y_test, y_pred)
        ev  += explained_variance_score(y_test, y_pred)
    result = {
        'Mean Absolute Error': mae / kf,
        'Mean Squared Error': mse / kf,
        'Max Error': me / kf,
        'R^2': r2 / kf,
        'Explained Variance': ev / kf,
    }
    return result

def run_model(model, X, y, kf):
    kfold = KFold(n_splits=kf, shuffle=True, random_state=42)
    y_tests, y_preds = [], []

    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        y_tests.append(y_test)
        y_preds.append(y_pred)

    return y_tests, y_preds

def run_models(models, X, y, kf):
    results = {}
    for name, model in models.items():
        print(f'Model {name} begining now ...')
        begin_time = time.time()

        y_tests, y_preds = run_model(model, X, y, kf)
        results[name] = evaluate_model(y_tests, y_preds, kf)

        end_time = time.time()
        duration = round((end_time - begin_time) / 60, 2)
        print(f'Model {name} run in'.ljust(50), f'{duration} minutes')

    results = pd.DataFrame(results).T
    results = results.reset_index()
    return results

def train_model(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    model.score(X_test, y_test)
    y_pred = model.predict(X_test)

    mae, mse, me, r2, ev = 0, 0, 0, 0, 0
    mae += mean_absolute_error(y_test, y_pred)
    mse += mean_squared_error(y_test, y_pred)
    me  += max_error(y_test, y_pred)
    r2  += r2_score(y_test, y_pred)
    ev  += explained_variance_score(y_test, y_pred)

    result = {
        'Mean Absolute Error': mae,
        'Mean Squared Error': mse,
        'Max Error': me,
        'R^2': r2,
        'Explained Variance': ev,
    }
    return result

def main():
    file_path_properties_2016 = "datasets/properties_2016.csv"
    file_path_train_2016 = "datasets/train_2016.csv"
    
    # load_properties(file_path_properties_2016)

    # show_missing_value(file_path_properties_2016)

    # analyse_missing_value(file_path_properties_2016)

    prop2016_df = load_data(file_path_properties_2016)
    prop2016 = pre_processing_data(prop2016_df)

    print('prop 2016 data has {0} rows and {1} columns'.format(prop2016.shape[0],prop2016.shape[1]))

    prop2016 = convert_data(prop2016)
    # prop2016.to_csv("results/properties_2016_proc.csv.gz", index=False, compression='gzip')
    prop2016 = transform_data(prop2016, prop2016_df)

    prop2016 = feature_engineering(prop2016)
    print ('prop 2016 data has {0} rows and {1} columns'.format(prop2016.shape[0],prop2016.shape[1]))
    print(prop2016.head())

    train2016_df = load_data(file_path_train_2016)
    train_2016 = convert_transactiondate(train2016_df.copy())

    train_2016 = train_2016.merge(prop2016, how='left', on='parcelid')
    print("train_2016 =", train_2016.head())
    print ('*** prop 2016 data has {0} rows and {1} columns'.format(prop2016.shape[0],prop2016.shape[1]))

    models = {
        'LinearRegression':          LinearRegression(),
        'DecisionTreeRegressor':     DecisionTreeRegressor(),
        'RandomForestRegressor':     RandomForestRegressor(n_estimators=100, random_state=42),
    }

    train_y = train_2016['logerror'].values
    train_x = train_2016.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2)

    for name, model in models.items():
        print(f'Model {name} begining now ...')
        begin_time = time.time()

        train_results = train_model(X_train, X_test, y_train, y_test, model)
        print(train_results)

        end_time = time.time()
        duration = round((end_time - begin_time) / 60, 2)
        print(f'Model {name} run in'.ljust(50), f'{duration} minutes')

    # train_results = run_models(models, train_x, train_y, 5)

    print('----------------------End-------------------------')

if __name__ == '__main__':
    # logerror=log(Zestimate)−log(SalePrice)
    main()
