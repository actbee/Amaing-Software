#include<opencv2/opencv.hpp> 
#include<iostream> 
#include<cmath>
#include "patch.h"
#define NUM 10
using namespace cv;
using namespace std;

const Scalar RED = Scalar(0, 0, 255);
const Scalar PINK = Scalar(230, 130, 255);
const Scalar BLUE = Scalar(255, 0, 0);
const Scalar LIGHTBLUE = Scalar(255, 255, 160);
const Scalar GREEN = Scalar(0, 255, 0);

struct POINT {
	int x=0;
	int y=0;
};

class System
{
public:
	enum {NOT_SET=0, IN_PROCESS=1, SET=2 };
	enum system_state{ZERO,ONE,TWO,THREE,FOUR,FIVE,SIX,SEVEN,EIGHT};
	void run(const Mat& _image, const string& _winName);
	void mouseClick(int event, int x, int y, int flags, void* param);
private:
	enum system_state current_state = ZERO;
	void showImage() const;
	void reset();
	void check_mask(Mat mask0);
	void setImageAndWinName(const Mat& _image, const string& _winName);
	void do_kmeans();
	bool do_grabcut();
	void do_patchmatch();
	void do_blank();
	void setRectInMask();
	void set_in_mask(int flags, Point p, bool BnoF);    //将用户标记的前景和背景写入mask里面
	const string* winName;
	const Mat* image; //对读入图像的指针
	Mat mask;  
	Mat withblank;
	uchar rectState;
	bool allow_select;
	uchar backset; //用户背景点标记状态
	uchar frontset; //用户前景点标记状态
	Rect rect;
	vector<Point> backpoints, frontpoints;
};

void System::do_patchmatch() {
	if (withblank.empty()) {
		cout << "error" << endl;
		return;
	}
	if (image->empty()) {
		cout << "error" << endl;
		return;
	}
	Mat res;
	image->copyTo(res);
	Mat patch_mask=Mat::zeros(res.size(),CV_8UC1);
	//Mat search = Mat(res.size(), CV_8UC3, Scalar::all(0));
	for (int row = 0; row < res.rows; row++) {
		for (int col = 0; col < res.cols; col++) {
			if (mask.at<uchar>(row, col) == GC_PR_FGD || mask.at<uchar>(row, col) == GC_FGD) {
				patch_mask.at<uchar>(row, col) = hole;
			}
			else {
				patch_mask.at<uchar>(row, col) = search;
			}
		}
	}
	Mat patch_mask0 = patch_mask.clone();
	Mat withblank0 = withblank.clone();
	int ps = 3;//空白像素修复窗口半径,即是patchsize
	int pyr = 5;//图像金字塔的层数
	int ann = 3;//最低最近最邻域迭代次数 anniteration
	for (int row = 0; row < patch_mask.rows; row++) {
		for (int col = 0; col < patch_mask.cols; col++) {
			if (patch_mask.at<uchar>(row, col) == hole) {
				int area = 3 * ps;  //空白像素窗口9*9
				int area_negx = col - area / 2 > 0 ? col - area / 2 : 0; //越界检查
				int area_posx = col + area / 2 < patch_mask.cols ? col + area / 2 : patch_mask.cols-1;
				int area_negy = row - area / 2 > 0 ?row - area / 2 : 0;
				int area_posy = row + area / 2 < patch_mask.rows ? row+ area / 2 : patch_mask.rows - 1;
				for (int i = area_negy; i < area_posy; i++) {   //对检索区域标记
					for (int j = area_negx; j < area_posx; j++) {
						patch_mask0.at<uchar>(i, j) = hole;
					}
				}
			}
		}
	}
	withblank0.setTo(0);
	for (int row = 0; row < patch_mask.rows; row++) {
		for (int col = 0; col < patch_mask.cols; col++) {
			if (patch_mask0.at<uchar>(row, col) == search) {
				withblank0.at<Vec3b>(row, col) = res.at<Vec3b>(row, col);
			}
		}
	}
	imshow("fill_blank", withblank0);
	imshow("mask", patch_mask);
	imshow("mask0", patch_mask0);
	waitKey(20);
	patch *patch_match = new patch(res, patch_mask0, ps, pyr, ann,"result.jpg");
    patch_match->Run();
	cout << "Now patch match finish@!!!" << endl;
	return;
}

void System::do_blank() {
	if (image->empty()) {
		cout << "error" << endl;
		return;
	}
	Mat res;
	image->copyTo(res);
	withblank = Mat::zeros(res.size(), CV_8UC3);
	for (int row = 0; row < res.rows; row++) {
		for (int col = 0; col < res.cols; col++) {
			if (mask.at<uchar>(row, col) == GC_BGD|| mask.at<uchar>(row, col) == GC_PR_BGD) {
				withblank.at<Vec3b>(row, col)[0] = res.at<Vec3b>(row, col)[0];
				withblank.at<Vec3b>(row, col)[1] = res.at<Vec3b>(row, col)[1];
				withblank.at<Vec3b>(row, col)[2] = res.at<Vec3b>(row, col)[2];
			}
			else {
				withblank.at<Vec3b>(row, col)[0] =255;
				withblank.at<Vec3b>(row, col)[1] =255;
				withblank.at<Vec3b>(row, col)[2] =255;
			}
		}
	}
	imshow("with_bank", withblank);
	return;
}

void System::check_mask(Mat mask0) {
	if (image->empty()) {
		cout << "error" << endl;
		return;
	}
	Mat res;
	image->copyTo(res);
	Mat result = Mat::zeros(res.size(), CV_8UC3);
	for (int row = 0; row < res.rows; row++) {
		for (int col = 0; col < res.cols; col++) {
			if (mask0.at<uchar>(row, col) == GC_PR_FGD) {
				result.at<Vec3b>(row, col)[0] = 255;
				result.at<Vec3b>(row, col)[1] = 255;
				result.at<Vec3b>(row, col)[2] = 255;
			}
			else if (mask0.at<uchar>(row, col) == GC_FGD) {
				result.at<Vec3b>(row, col)[0] = 255;
				result.at<Vec3b>(row, col)[1] = 155;
				result.at<Vec3b>(row, col)[2] = 25;
			}
			else if (mask0.at<uchar>(row, col) == GC_PR_BGD) {
				result.at<Vec3b>(row, col)[0] = 155;
				result.at<Vec3b>(row, col)[1] = 255;
				result.at<Vec3b>(row, col)[2] = 125;
			}
		}
	}
	imshow("check", result);
}

void System::set_in_mask(int flags, Point p, bool BnoF) {
	vector<Point> *front=&frontpoints;
	vector<Point> *background = &backpoints;
	if (BnoF) {
		background->push_back(p);
		circle(mask, p, 2, GC_BGD, -1);
	}
	else {
		front->push_back(p);
		circle(mask, p, 2, GC_FGD, -1);
	}
}

bool System::do_grabcut() {
	if (image->empty()) {
		cout << "error" << endl;
		return false;
	}
	Mat res, bgModel, fgModel;
	image->copyTo(res);
	Mat mask0;

	cout << "grabcut is now doing....";

	if (current_state == THREE) {
		mask0=Mat::zeros(mask.size(), CV_8UC1);
		for (int row =rect.y; row < rect.y+rect.height; row++) {
			for (int col = rect.x; col < rect.x+rect.width; col++) {
				mask0.at<uchar>(row, col) = mask.at<uchar>(row, col);
			}
		}
		check_mask(mask0);
		waitKey(10);
		for (int i = 0; i < 5; i++) {
			grabCut(res, mask0, rect, bgModel, fgModel, 1, GC_EVAL);
			cout << " ... ";
		}
		//对选区图像进行操作
		Mat foreground1(res.size(), CV_8UC3, Scalar(255, 255, 255));
		for (int row = rect.y; row < rect.y + rect.height; row++) {
			for (int col =rect.x; col < rect.x+rect.width; col++) {
				if (mask0.at<uchar>(row, col) == GC_PR_FGD || mask0.at<uchar>(row, col) == GC_FGD) {
					foreground1.at<Vec3b>(row, col) = res.at<Vec3b>(row, col);
					mask0.at<uchar>(row, col) = GC_FGD;
				}
			}
		}
		imshow("selected area objects", foreground1);
		cout << endl<<"only do the rect! " << rect.x << " , " << rect.y << " , "<<rect.width<<" , "<<rect.height<<endl;
		waitKey(10);
		cout << endl << "grabcut finished!" << endl;
	}

	else if (current_state = FOUR) {
		mask0 = mask.clone();
		Rect rec0(0, 0, res.cols, res.rows);
		for (int i = 0; i < 5; i++) {
			grabCut(res, mask0, rec0, bgModel, fgModel, 1);
			cout << " ... ";
		}
		//对全局图像的处理显示，寻找类似物体
		Mat foreground2(res.size(), CV_8UC3, Scalar(255, 255, 255));
		Mat mask1;
		compare(mask0, GC_PR_FGD, mask1, CMP_EQ); //保留mask0部分的前景区
		//将原图像src中的result区域拷贝到foreground中
		res.copyTo(foreground2, mask1);
		Mat mask2;
		compare(mask, GC_FGD, mask2, CMP_EQ); //保留mask0部分的前景区
		res.copyTo(foreground2, mask2);
		rectangle(foreground2, Point(rec0.x, rec0.y), Point(rec0.x + rec0.width, rec0.y + rec0.height), RED, 2);
		imshow("grabcut", foreground2);
		cout << endl << "grabcut finished!" << endl;
	}

	check_mask(mask0);
	cout << "Do you want to redo it? Please enter ‘n' to continue without redo. Enter others to redo." << endl;
	int c = waitKey(0);
		if ((char)c == 'n') {
			if (current_state == FOUR) {
				mask = mask0;
			}
			else if (current_state == THREE) {
				//mask0.copyTo(mask(rect));
				for (int row =rect.y; row < rect.y+rect.height; row++) {
					for (int col =rect.x; col < rect.x+rect.width; col++) {
						mask.at<uchar>(row , col) = mask0.at<uchar>(row, col);
					}
				}
			}
		}
		else {
			cout << "ok, redo!" << endl;
			return false;
		}
		check_mask(mask);
		return true;
}

void System::do_kmeans() {
	if (image->empty()) {
		return;
	}
	Mat res;  //原图像
	image->copyTo(res);
	Mat roi = res(rect); //roi 存储了框选的选区图像
	imshow("interest area", roi);

	int width = roi.cols;
	int height = roi.rows;
	int dims = roi.channels();
	int sampleCount = width * height;
	int clusterCount = NUM;   //聚类数量

	Mat points(sampleCount, dims, CV_32F, Scalar(10)); //创建一个图像，每一个像素dims个10，sampleCount维
	Mat labels; //接受k-means后每个像素的归类，形状1*（height*weight）
	Mat centers(clusterCount, 1, points.type());  //用来存储聚类后的中心点

	int index = 0;
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			index = row * width + col;
			Vec3b lab = roi.at<Vec3b>(row, col);  //原始图像的lab
			points.at<float>(index, 0) = static_cast<int>(lab[0]);
			points.at<float>(index, 1) = static_cast<int>(lab[1]);
			points.at<float>(index, 2) = static_cast<int>(lab[2]);
		}
	}
	//cvtColor(points, points, COLOR_BGR2Lab);
	// 运行K-Means数据分类 
	TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0); //迭代算法的终止条件（参数类型，迭代最大次数，特定阈值）
	kmeans(points, clusterCount, labels, criteria, 10, KMEANS_PP_CENTERS, centers);  //use k-means 
	
	cout << centers.rows << "  " << centers.cols << endl;
	/*
	//聚类中心的显示（色彩为标准聚类，不是位置坐标）
    for (int i = 0; i < NUM; i++) {
		cout << centers.at<float>(i,0)<< ","<<centers.at<float>(i,1)<<","<<centers.at<float>(i,2)<<endl;
	}
	*/
	// 显示图像分割结果 
	Mat results[NUM];
	for (int i = 0; i < NUM; i++) {
		results[i] = Mat::zeros(roi.size(), CV_8UC3);
	}
	int total[NUM];
	for (int i = 0; i < NUM; i++) {
		total[i] = 0;
	}
	for (int row = 0; row < height; row++) {  //对结果的每个图像根据归类结果上色
		for (int col = 0; col < width; col++) {
			index = row * width + col;
			int label = labels.at<int>(index, 0);
			total[label]++;
			results[label].at<Vec3b>(row, col)[0] = 255;
			results[label].at<Vec3b>(row, col)[1] = 255;
			results[label].at<Vec3b>(row, col)[2] = 255;
		}
	}
	cout << "Do you want to see the results of K-means? type in 'Y/y' if you want, other inputs will be regarded as no. " << endl;
	char see;
	cin >> see;
	if (see=='Y'||see=='y') {
		for (int i = 0; i < NUM; i++) {
			imshow(to_string(i), results[i]);
		}
	}

	/* // 将分割结果显示在一张图上
	Mat result = Mat::zeros(roi.size(), CV_8UC3);
	int total[NUM];
	for (int i = 0; i < NUM; i++) {
		total[i] = 0;
	}
	for (int row = 0; row < height; row++) {  //对结果的每个图像根据归类结果上色
		for (int col = 0; col < width; col++) {
			index = row * width + col;
			int label = labels.at<int>(index, 0);
			total[label ]++;
			if (label > 0 && label <= 5) { 
				result.at<Vec3b>(row, col)[0] = (255 / clusterCount)*label;
				result.at<Vec3b>(row, col)[1] = 0;
				result.at<Vec3b>(row, col)[2] = 30 + label * 10;
			}
			else if (label > 5 && label <= 10) {
				result.at<Vec3b>(row, col)[0] = 0;
				result.at<Vec3b>(row, col)[1] = 60 + label * 10;
				result.at<Vec3b>(row, col)[2] = 30;
			}
			else if (label > 10) {
				result.at<Vec3b>(row, col)[0] = 70;
				result.at<Vec3b>(row, col)[1] = 0;
				result.at<Vec3b>(row, col)[2] = (255 / clusterCount)*label;
			}

			else if (label == 0) {
				result.at<Vec3b>(row, col)[0] = 0;
				result.at<Vec3b>(row, col)[1] = 0;
				result.at<Vec3b>(row, col)[2] = 0;
			}
		}
	}
	imshow("kmeans", result);
	*/
	
	std::cout << "the result is as follows" << std::endl;
	for (int i = 0; i < NUM; i++) {
		std::cout << i << " has " << total[i] << std::endl;
	}
	std::cout << "k-means method has finished" << std::endl;
	cout << endl;
	cout << "now decide which part is front and which part is background..." << endl;

//遍历边缘寻找背景组
	Mat backorfront=labels.clone(); //backorfront模板，0表示背景，1表示	前景，-1表示未设置
	
	int back_collector[NUM]; //被采集到的边缘对应的lable记录为1
 	for (int i = 0; i < NUM; i++) {
		back_collector[i] = 0;
	}
		for (int row=0; row < height; row++) {
			for (int col = 0;col< width; col++) {
				index = row * width + col;
				backorfront.at<int>(index,0) = -1;
			}
	}
		bool check_xleft = true;
		bool check_xright = true;
		bool check_yup = true;
		bool check_ydown = true;
		cout << "Anything you do not want to check? Enter '1' means yes, otherwise no." << endl;
		int k = waitKey(0);
		if ((char)k == '1') {
			cout << "check_xleft?" << endl;
			int xl = waitKey(0);
			if ((char)xl == '0') {
				check_xleft = false;
				cout << "no" << endl;
			}
			else {
				cout << "yes" << endl;
			}
			cout << "check_xright?" << endl;
			int xr = waitKey(0);
			if ((char)xr =='0') {
				check_xright = false;
				cout << "no" << endl;
			}
			else {
				cout << "yes" << endl;
			}
			cout << "check_ydown?" << endl;
			int yd = waitKey(0);
			if ((char)yd == '0') {
				check_ydown = false;
				cout << "no" << endl;
			}
			else {
				cout << "yes" << endl;
			}
			cout << "check_yup?" << endl;
			int yu = waitKey(0);
			if ((char)yu == '0') {
				check_yup = false;
				cout << "no" << endl;
			}
			else {
				cout << "yes" << endl;
			}
		}
		else {
			cout << "ok, nothing special!" << endl;
		}
		for (int row = 0; row < height; row++) {
			// index = row * width + col;
			if (check_xleft == true) {
				int lable1 = labels.at<int>(row*width, 0);
				back_collector[lable1] = 1;
			}
			if (check_xright == true) {
				int lable2 = labels.at<int>(row*width + width - 1, 0);
				back_collector[lable2] = 1;
			}
	}
		for (int col = 0; col < width; col++) {
			if (check_yup == true) {
				int lable3 = labels.at<int>(col, 0);
				back_collector[lable3] = 1;
			}
			if (check_ydown == true) {
				int lable4 = labels.at<int>((height - 1)*width + col, 0);
				back_collector[lable4] = 1;
			}
		}
	
//寻找前景
		/*
		//方法一：根据聚类中心遍历图像寻找最接近像素点的物理位置，然后计算这些物理物质中离边缘组最远的类为前景
		POINT p[NUM];
		for (int i = 0; i < NUM; i++) {
			float mindis = 9999999;
			for (int row = 0; row < height; row++) {
				for (int col = 0; col < width; col++) {
					float r = abs(static_cast<float>(res.at<Vec3b>(row, col)[0]) - centers.at<float>(i, 0));
					float g = abs(static_cast<float>(res.at<Vec3b>(row, col)[1]) - centers.at<float>(i, 1));
					float b= abs(static_cast<float>(res.at<Vec3b>(row, col)[2]) - centers.at<float>(i, 2));
					float dis = r + g + b;
					if (dis < mindis) {
						mindis = dis;
						p[i].x = col;
						p[i].y = row;
					}
				}
			}
		}
		
		cout << "the result of each centers: " << endl;
		for (int i = 0; i < NUM; i++) {
			cout << p[i].x << "," << p[i].y << "   ||||    ";
		}
		cout << endl;
		
		float max = 0;
		int front_id = -1;;
		for (int i = 0; i < NUM; i++) {
			if (back_collector[i] == 0) {
				float sum = 0;
				for (int j = 0; j < NUM; j++) {
					if (back_collector[j] == 1) {
						sum += sqrt(pow((p[i].x-p[j].x),2) + pow((p[i].y- p[j].y), 2));
					}
				}
				if (sum >max) {
					max = sum;
					front_id = i;
				}
			}
		}
		if (front_id == -1) {
			cout << "seems no front lyer is determined" << endl;
		}
		else {
			back_collector[front_id] = 2;
		}
		*/
		//方法二：算每个聚类中心离其他颜色欧氏距离最远的类作为前景
		float max = 0;
		int front_id = -1;;
		for (int i = 0; i < NUM; i++) {
			if (back_collector[i] == 0) {
				float sum = 0;
				for (int j = 0; j < NUM; j++) {
					if (back_collector[j] == 1) {
						sum += sqrt(pow((centers.at<float>(i, 0) - centers.at<float>(j, 0)), 2) + pow((centers.at<float>(i, 1) - centers.at<float>(j, 1)), 2) + pow((centers.at<float>(i, 2) - centers.at<float>(j,2)), 2));
					}
				}
				if (sum > max) {
					max = sum;
					front_id = i;
				}
			}
		}
		if (front_id == -1) {
			cout << "seems no front lyer is determined" << endl;
		}
		else {
			back_collector[front_id] = 2;
		}

		//输出每个类别的判断，0表示可能前景，1表示背景，2表示前景
		for (int i = 0; i < NUM; i++) {
			cout << back_collector[i] << " ";
		}
		cout << endl;


		for (int row = 0; row < height; row++) {
			for (int col = 0; col < width; col++) {
				int index = row * width + col;
				if (back_collector[labels.at<int>(index,0)] == 1) {
					backorfront.at<int>(index, 0) = 0;
				}
				else if (back_collector[labels.at<int>(index, 0)] == 2) {
					backorfront.at<int>(index, 0) = 1;
				}
			}
		}

		//显示下分类后初步判断的结果
		Mat check_result = Mat::zeros(roi.size(), CV_8UC1);
		for (int row = 0; row < height; row++) {
			for (int col = 0; col < width; col++) {
				int index = row * width + col;
				if (backorfront.at<int>(index,0) == 0) {
					check_result.at<uchar>(row, col)= 255;
				   mask.at<uchar>(rect.y+row, rect.x+col) = GC_BGD; //mask遮罩进行对应背景标记
				}
				
				else if (backorfront.at<int>(index, 0) == 1) {
					check_result.at<uchar>(row, col) = 150;
					mask.at<uchar>(rect.y + row, rect.x + col) = GC_FGD; //mask遮罩进行对应前景标记
				}
				else {
					check_result.at<uchar>(row, col) = 200;
					mask.at<uchar>(rect.y + row, rect.x + col) = GC_PR_FGD; //mask遮罩进行对应前景标记
				}
			}
		}
		imshow("backorfront", check_result);	
		check_mask(mask);
}

void System::run(const Mat& src, const string& winName) {
	for (;;) {
	key_change:
		switch (current_state) {
		case ZERO: {
			cout << "AMAZING SOFTWARE___TEST VERSION" << endl;
			cout << "At any time, enter 'r' to reset all and press 'esc' to exit." << endl << endl;
			setImageAndWinName(src, winName);
			showImage();
			current_state = ONE;
			break;
		}
		case ONE: {
			cout << endl << "Stage ONE: please select one area with the objects you want to make animate." << endl;
			cout << "Use your mouse to select area. Enter 'n' to confirm...." << endl << endl;
			while (1) {
				int c = waitKey(0);
				switch ((char)c) {
				case 'n': {
					if (rectState == SET) {
						current_state = TWO;
						cout << "Area has comfirmed." << endl;
						goto key_change;
						break;
					}
					else {
						cout << "Please select the are first! " << endl;
						break;
					}
					break;
				}
				case'\x1b': {
					cout << "Exiting---" << endl;
					return;
					break;
				}
				case 'r': {
					cout << endl;
					reset();
					showImage();
					goto key_change;
					break;
				}
				default: {
					cout << (char)c << " is no use. Please enter the right key.^--^" << endl;
					break;
				}
				}
			}
			break;
		}
		case TWO: {
			cout << endl << "Stage TWO: K-MEANS is used in the seleced area..." << endl << endl;
			do_kmeans();
			cout << " enter 'n' to go to the next stage. " << endl;
			while (1) {
				int c = waitKey(0);
				switch ((char)c) {
				case'\x1b': {
					cout << "Exiting---" << endl;
					return;
				}
				case 'r': {
					cout << endl;
					reset();
					showImage();
					goto key_change;
					break;
				}
				case 'n': {
					current_state = THREE;
					goto key_change;
					break;
				}
				default: {
					cout << (char)c << " is no use. Please enter the right key.^--^" << endl;
					break;
				}
				}
			}
		}
		case THREE: {
			cout << endl << "Stage THREE: Extract the object in the selected area..." << endl;
			cout << endl << "Now please mark the certain back layer with the right mouse buttom and the certain front layer with the left mouse buttom." << endl;
			cout << endl << "The marked back layer will be shown as blue while the front back layer as red." << endl;
			cout << "You may enter 'n' when finished." << endl;
			allow_select = true;
			check_mask(mask);
			while (1) {
				int c = waitKey(0);
				switch ((char)c) {
				case'\x1b': {
					cout << "Exiting---" << endl;
					return;
				}
				case 'n': {
					allow_select = false;
					frontpoints.clear();
					backpoints.clear();
					cout << "Now the mark part is finished!" << endl;
					cout << "And Grabcut is used to catch the object in the area out..." << endl << endl;
					if (!do_grabcut()) {
						current_state = THREE;
					}
					else {
						current_state = FOUR;
					}
					goto key_change;
					break;
				}
				case 'r': {
					cout << endl;
					reset();
					showImage();
					goto key_change;
					break;
				}
				default: {
					cout << (char)c << " is no use. Please enter the right key.^--^" << endl;
					break;
				}
				}
			}
		}
		case FOUR: {
			cout << endl << "Stage FOUR: Find the similar objects in the whole picture using grabcut..." << endl;
			cout << " Now choose the certain front objects using left mouse buttom and the certain background using right mouse buttom just like Stage THREE."<<endl<<
				"You may enter 'n' once finished. " << endl;
			allow_select = true;
			while (1) {
				int c = waitKey(0);
				switch ((char)c) {
				case'n': {
					cout << "another Grabcut is used here to get more details..." << endl;
					allow_select = false;
					frontpoints.clear();
					backpoints.clear();
					if(!do_grabcut()){
						current_state = FOUR;
					}
					else {
						current_state = FIVE;
					}
					goto key_change;
					break;
				}
				case'\x1b': {
					cout << "Exiting---" << endl;
					return;
				}
				case 'r': {
					cout << endl;
					reset();
					showImage();
					goto key_change;
					break;
				}
				default: {
					cout << c << "is no use. Please enter the right key.^--^" << endl;
					break;
				}
				}
			}

		}
		case FIVE: {
			cout << endl << "Stage FIVE: Patchmatch is used to fix the blank areas..." << endl;
			cout << "Now the picture has many blanks." << endl;
			do_blank();
			cout << "Please enter 'n' and let's do Patchmatch. ^~~^" << endl;
			while (1) {
				int c = waitKey(0);
				switch ((char)c) {
				case'\x1b': {
					cout << "Exiting---" << endl;
					return;
				}
				case 'n': {
					do_patchmatch();
					current_state = SIX;
					goto key_change;
					break;
				}
				case 'r': {
					cout << endl;
					reset();
					showImage();
					goto key_change;
					break;
				}
				default: {
					cout <<(char)c << " is no use. Please enter the right key.^--^" << endl;
					break;
				}
				}
			}
			break;
		}
		case SIX: {
			cout << endl << "Stage SIX: Now choose the order of layers..." << endl;
			while (1) {
				int c = waitKey(0);
				switch ((char)c) {
				case'\x1b': {
					cout << "Exiting---" << endl;
					return;
				}
				case 'r': {
					cout << endl;
					reset();
					showImage();
					goto key_change;
					break;
				}
				default: {
					cout << (char)c << " is no use. Please enter the right key.^--^" << endl;
					break;
				}
				}
			}
			break;
		}
		}
	}
}

void System::reset() {
	if (!mask.empty()) {
		mask.setTo(Scalar::all(GC_PR_BGD));
	}
	backpoints.clear();
	frontpoints.clear();
	allow_select = false;
	rectState = NOT_SET;
	backset = NOT_SET;
	frontset = NOT_SET;
	current_state = ONE;
}

void System::setImageAndWinName(const Mat& _image, const string& _winName) {
	if (_image.empty() || _winName.empty()) {
		return;
	}
	image = &_image;
	winName = &_winName;
	mask.create(image->size(), CV_8UC1);
	reset();
}

void System::setRectInMask() { //选中区域确定后设置对应区域的mask
	assert(!mask.empty());
	mask.setTo(GC_PR_BGD);
	rect.x = max(0, rect.x);
	rect.y = max(0, rect.y);
	rect.width = min(rect.width, image->cols - rect.x);
	rect.height = min(rect.height, image->rows - rect.y);
	(mask(rect)).setTo(Scalar(GC_PR_FGD));//mask的rect区域的所有像素全部设置为可能的前景区
}

void System::showImage() const {
	if (image->empty() || winName->empty()) {
		return;
	}
	Mat res;
	image->copyTo(res);
	if (rectState == IN_PROCESS || rectState == SET) {
		rectangle(res, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), RED, 2);
	}
	if(current_state==THREE)
	vector<Point>::const_iterator it;
	for (auto it = backpoints.begin(); it != backpoints.end(); ++it) {
		circle(res, *it, 2, BLUE,-1); //画圆 参数是对象图片，点的位置，半斤大小，颜色，厚度
	}
	for (auto it = frontpoints.begin(); it != frontpoints.end(); ++it) {
		circle(res, *it, 2, RED, -1);
	}
	imshow(*winName, res);
}

void System::mouseClick(int event, int x, int y, int flags, void*) {
	if (allow_select==true) {
		switch (event) {
		case CV_EVENT_LBUTTONDOWN:
		{
			frontset = IN_PROCESS;
			break;
		}
		case CV_EVENT_RBUTTONDOWN: 
		{
			backset = IN_PROCESS;
			break;
		}
		case CV_EVENT_LBUTTONUP:
		{
			if (frontset == IN_PROCESS) {
				set_in_mask(flags, Point(x, y), false);  //画出前景点
			    frontset = SET;
				showImage();
			}
			break;
		}
		case CV_EVENT_RBUTTONUP:
		{
			if (backset == IN_PROCESS) {
				set_in_mask(flags, Point(x, y), true); // 画出背景点
				backset = SET;
				showImage();
             }
			 break;
		}
		case CV_EVENT_MOUSEMOVE: {
			if (frontset == IN_PROCESS) {
				set_in_mask(flags, Point(x, y), false);
				showImage();
			}
			else if (backset == IN_PROCESS) {
				set_in_mask(flags, Point(x, y), true);
				showImage();
			}
			break;
		}
		}
		return;
	}
	else if (current_state == ONE) {
		switch (event) {
		case CV_EVENT_LBUTTONDOWN:
		{
			rectState = IN_PROCESS;
			rect = Rect(x, y, 1, 1);
			break;
		}
		case CV_EVENT_LBUTTONUP:
		{
			if (rectState == IN_PROCESS) {
				rect = Rect(Point(rect.x, rect.y), Point(x, y));
				rectState = SET;
				setRectInMask();
				showImage();
			}
			break;
		}
		case CV_EVENT_MOUSEMOVE:
		{
			if (rectState == IN_PROCESS) {
				rect = Rect(Point(rect.x, rect.y), Point(x, y));
				showImage();
			}
			break;
		}
		}
		return;
	}
}

System myapp;

static void on_mouse(int event, int x, int y, int flags, void* param) {
	myapp.mouseClick(event, x, y, flags, param);
}

int main() {
	string filename = "E:\\Uni\\SIGGRAPH\\UIST\\pics\\nishiripoli.jpg";
	//Mat src0 = imread(filename,COLOR_RGB2Lab); //用Lab通道打开图片
	Mat src0 = imread(filename);
	/*检查图像
	cout << "the LAB color space: " << endl;
	for (int i = 0; i < src0.cols; i++) {
		cout << (int)src0.at<Vec3b>(0, i)[0] << " , " << (int)src0.at<Vec3b>(0, i)[1] << " , " << (int)src0.at<Vec3b>(0, i)[2] << "   ";
	}
	cout << endl;
	cout << "the RGB color space: " << endl;
	for (int i = 0; i < src1.cols; i++) {
		cout << (int)src1.at<Vec3b>(0, i)[0] << " , " << (int)src1.at<Vec3b>(0, i)[1] << " , " << (int)src1.at<Vec3b>(0, i)[2] << "   ";
	}
	cout << endl;
	*/
	//Mat src0 = imread(filename);
	
	/* 对图片的LAB值映射到正常范围
	for (int i = 0; i < src0.cols; i++) {
		for (int j = 0; j < src0.rows; j++) {
		
			src0.at<Vec3b>(j, i)[0]=src0.at<Vec3b>(j, i)[0] * 100 / 255;
			src0.at<Vec3b>(j, i)[1] -= 128;
			src0.at<Vec3b>(j, i)[2] -= 128;
			
			cout << (int)src0.at<Vec3b>(j, i)[0] << "," <<(int) src0.at<Vec3b>(j, i)[1] << "," << (int)src0.at<Vec3b>(j, i)[2] << "   ";
		}
	}
	*/
	if (src0.empty()) {
		cout << "no such image named " << filename <<endl;
		return -1;
	}
	Mat src;
	resize(src0, src, Size(), 0.1, 0.1);  //缩小图片到src

	const string winName = "image";

	cvNamedWindow(winName.c_str(), CV_WINDOW_AUTOSIZE);
	cvSetMouseCallback(winName.c_str(), on_mouse, 0);

	//myapp.setImageAndWinName(src, winName);
	
	myapp.run(src, winName);

	cvDestroyWindow(winName.c_str());

	
	return 0;
}