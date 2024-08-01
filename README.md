# Hello-world
first day in github
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <map>
#include <random>
#include <stdexcept>
#include <iomanip>

using namespace std;

// 创建Node类，关联数据集与标记
struct Node {
    vector<long double> data;
    int label;
};
//创建Data类，实现数据的读取和输出
class Data {
private:
    vector<Node> dataset;

public:
    // 读取数据
    void readData(const string& address) {
        ifstream ifile(address);
        if (!ifile.is_open()) {
            cerr << "文件打开错误！！！ " << address << endl;
            return;
        }
        string line;
        dataset.clear();//清除前一次的数据，否则输入不同的数据会报错越界
        while (getline(ifile, line)) {
            stringstream ss(line);
            long double val;
            vector<long double> temp;
            while (ss >> val) {
                temp.push_back(val);
            }
            //异常处理
            try {
                if (temp.empty()) {
                    throw runtime_error("数据行为空");
                }
                int label = static_cast<int>(temp.back());  
                temp.pop_back();
                dataset.push_back({ temp, label });
            }
            catch (const bad_cast& error) {
                cerr << "错误：最后一个元素不是整数" << endl;
                cerr << "异常信息为：" << error.what() << endl;
            }
            catch (const exception& error) {
                cerr << "错误：数据解析过程中发生异常。" << endl;
                cerr << "异常信息为: " << error.what() << endl;
            }    
        }
        ifile.close();
    }

    // 显示数据
    void showData() const {
        for (const auto& node : dataset) {
            cout << "Label: " << node.label << endl;
            cout << "Features: ";
            for (long double feature : node.data) {
                cout << left<<setw(5)<<feature<<" ";
            }
            cout << endl << endl;
        }
    }
    //用以调取dataset的函数，便于将数据赋值给其他类的变量进行运算
    const vector<Node>& getDataset() const {
        return dataset;
    }
};
//创建KNN类，实现KNN的估计、评估功能
class KNN {
private:
    vector<Node> dataset;

    // 计算欧式距离
    double calc_O(const Node& n1, const Node& n2) const {
        double distance = 0.0;
        for (size_t i = 0; i < n1.data.size(); i++) {
            distance += pow(n1.data[i] - n2.data[i], 2);
        }
        return sqrt(distance);
    }

    // 计算曼哈顿距离
    double calc_M(const Node& n1, const Node& n2) const {
        double distance = 0.0;
        for (size_t i = 0; i < n1.data.size(); i++) {
            distance += abs(n1.data[i] - n2.data[i]);
        }
        return distance;
    }

public:
    KNN(const vector<Node>& data) : dataset(data) {}

    // 预测
    int predict(const vector<long double>& test_data, int k, int cal) const {
        vector<pair<double, int>> distances;
        for (const auto& train_node : dataset) {
            double distance = 0.0;
            switch (cal) {
            case 1:
                distance = calc_O(train_node, { test_data, -1 });
                break;
            case 2:
                distance = calc_M(train_node, { test_data, -1 });
                break;
            }
            distances.push_back({ distance, train_node.label });
        }
        sort(distances.begin(), distances.end());

        map<int, int> class_freq;
        for (int i = 0; i < k; ++i) {
            int label = distances[i].second;
            class_freq[label]++;
        }

        int max_freq = 0;
        int result_label = -1;
        for (const auto& pair : class_freq) {
            if (pair.second > max_freq) {
                max_freq = pair.second;
                result_label = pair.first;
            }
        }
        return result_label;
    }

    // 评估模型
    vector<double> evaluate(double train_ratio, int k) const {
        vector<Node> train_data;
        vector<Node> test_data;
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(0, 1);

        for (const auto& node : dataset) {
            if (dis(gen) < train_ratio) {
                train_data.push_back(node);
            }
            else {
                test_data.push_back(node);
            }
        }

        int correct_O = 0;
        int correct_M = 0;

        for (const auto& test_node : test_data) {
            int predicted_label_O = predict(test_node.data, k, 1);
            int predicted_label_M = predict(test_node.data, k, 2);

            if (predicted_label_O == test_node.label) correct_O++;
            if (predicted_label_M == test_node.label) correct_M++;
        }

        double accuracy_O = static_cast<double>(correct_O) / test_data.size();
        double accuracy_M = static_cast<double>(correct_M) / test_data.size();

        return { accuracy_O, accuracy_M };
    }
};

// 创建Menu类
class Menu {
private:
    Data temp_set;

public:
    // 生成交互菜单
    void showMenu() const {
        cout << "**************" << endl;
        cout << "1.显示数据信息" << endl;
        cout << "2.评估模型" << endl;
        cout << "3.预测推理" << endl;
        cout << "4.退出程序" << endl;
        cout << "**************" << endl;
    }

    // 获取选项输入，程序功能实现
    void run() {
        while (true) {
            showMenu();
            int choice;
            cout << "请输入您的选择：";
            if (!(cin >> choice)) {
                cout << "输入无效，请输入一个整数。" << endl;
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
                continue;
            }
            string address;
            switch (choice) {
            case 1:
                cout << "请输入数据集地址：";
                cin >> address;
                temp_set.readData(address);
                temp_set.showData();
                break;
            case 2: {
                cout << "请输入数据集地址：";
                cin >> address;
                temp_set.readData(address);

                double ratio;
                int k;
                cout << "请输入训练集占比：";
                cin >> ratio;
                cout << "请输入k值：";
                cin >> k;

                KNN knn(temp_set.getDataset());
                vector<double> acc = knn.evaluate(ratio, k);
                cout << "若采用欧式距离，正确率为：" << acc[0] * 100 << "%" << endl;
                cout << "若采用曼哈顿距离，正确率为：" << acc[1] * 100 << "%" << endl;
                break;
            }
            case 3: {
                cout << "请输入数据集地址：";
                cin >> address;
                temp_set.readData(address);

                cout << "请输入测试的数据（用空格分隔）：";
                cin.ignore();
                string line;
                getline(cin, line);
                stringstream ss(line);
                vector<long double> test_data;
                long double item;
                while (ss >> item) {
                    test_data.push_back(item);
                }
                func:
                int k, cal;
                cout << "请输入k值：";
                if (!(cin >> k)) {
                    cout << "输入无效，请输入一个整数。" << endl;
                    cin.clear();
                    cin.ignore(numeric_limits<streamsize>::max(), '\n');
                    continue;
                }
                while (true) {
                    cout << "若计算欧式距离，请输入1，若计算曼哈顿距离，请输入2：";
                    if (!(cin >> cal)) {
                        cout << "输入无效，请输入一个整数。" << endl;
                        cin.clear();
                        cin.ignore(numeric_limits<streamsize>::max(), '\n');
                        continue;
                    }
                    if (cal != 1 && cal != 2) {
                        cout << "输入无效，请输入1或2。" << endl;
                        continue; // 继续下一次循环
                    }
                    break;
                }
                try {
                    if (test_data.size() != temp_set.getDataset()[0].data.size()) {
                        throw runtime_error("测试数据维度与训练数据维度不匹配");
                    }
                    KNN knn(temp_set.getDataset());
                    int predicted_label = knn.predict(test_data, k, cal);
                    cout << "预测的类型为：" << predicted_label << endl;
                    break; // 如果没有异常，跳出循环
                }
                catch (const runtime_error& error) {
                    cerr << "错误: " << error.what() << endl;
                    cout << "请重新输入测试数据（用空格分隔）：" << endl;
                    cin.ignore();
                    string line;
                    getline(cin, line);
                    stringstream ss(line);
                    test_data.clear(); // 清空之前输入的数据
                    long double item;
                    while (ss >> item) {
                        test_data.push_back(item);
                    }
                    goto func;
                }
            }
            case 4:
                return;
            default:
                cout << "数字不符合要求，请重新输入：" << endl;
            }

        }
    };
};
int main() {
    Menu menu;
    menu.run();
    return 0;
}
