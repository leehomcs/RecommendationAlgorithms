import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def sparseFeature(feat, feat_onehot_dim, embed_dim):
    return {'feat': feat, 'feat_onehot_dim': feat_onehot_dim, 'embed_dim': embed_dim}

def denseFeature(feat):
    return {'feat': feat}

def data_preprocessing(data, embed_dim=8, test_size=0.2):
    cols = data.columns.values
    dense_feats = [f for f in cols if f[0] == 'I']
    categorical_feats = [f for f in cols if f[0] == 'C']

    data[dense_feats] = data[dense_feats].fillna(0)
    data[categorical_feats] = data[categorical_feats].fillna('-1')

    # 归一化
    data[dense_feats] = MinMaxScaler().fit_transform(data[dense_feats])
    # LabelEncoding编码
    for col in categorical_feats:
        data[col] = LabelEncoder().fit_transform(data[col])

    feature_columns = [[denseFeature(feat) for feat in dense_feats]] + \
                      [[sparseFeature(feat, data[feat].nunique(), embed_dim) for feat in categorical_feats]]

    # 数据集划分
    X = data.drop(['label'], axis=1).values
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    return feature_columns, (X_train, y_train), (X_test, y_test)


def criteo_reader():
    data = pd.read_csv('/Users/leehom/PycharmProjects/RecommendationAlgorithms/Datasets/criteo/criteo_sample.csv')
    # print(data.shape)
    preprocessed_data = data_preprocessing(data, embed_dim=8, test_size=0.2)
    # print(preprocessed_data.shape)
    return preprocessed_data


if __name__ == '__main__':
    data = criteo_reader()