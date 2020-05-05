#include<opencv2/opencv.hpp>
#include<cv.h>
#include <iostream>
#include<vector>
#include "opencv2/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include<math.h>
//#include<windows.h>

using namespace std; 
using namespace cv;

int c;


double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0) //求角度
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}







int redThre =49; // 115~135  
int saturationTh = 7; //55~65  
Mat CheckColor(Mat &inImg);
void DrawFire(Mat &inputImg, Mat foreImg);
 
int main()
{
	VideoCapture capture("IMG_2223.mp4");//input the video
    
	while (1)
	{
		Mat frame;
		
		capture >> frame;
		if (frame.empty())
			break;



//    HANDLE hcom;// serial communication
//    hcom = CreateFile("COM3",GENERIC_READ | GENERIC_WRITE,0,NULL,OPEN_EXISTING 
//                       ,FILE_ATTRIBUTE_NORMAL,NULL);
//    if (hcom == INVALID_HANDLE_VALUE)
//    {
       
//		fprintf(stderr, "打开串口失败！\n");
//        exit(0);
//    }
//    SetupComm(hcom,1024,1024);
//    DCB dcb;
//    GetCommState(hcom,&dcb);
//    dcb.BaudRate = 9600;
//    dcb.ByteSize = 8;
//    dcb.Parity = 0;
//    dcb.StopBits = 1;
//   SetCommState(hcom,&dcb);
//    char data[]={0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
//    DWORD dwWrittenLen = 0;
//	int k=0;
   



cv::Mat srcImg, tempImg,gary,thresh,srcImg1;
    cv::Mat dstImg; 
    int j = 0;
    CvFont font;
    double s = 0;
    srcImg = frame;
    if (!srcImg.data)
    {
        std::cout << "no img" << std::endl;
        return -1;
    }
    //图片处理
    resize(srcImg, srcImg, cv::Size(srcImg.cols / 2, srcImg.rows / 2), 0, 0);//缩放
    cvtColor(srcImg, gary, CV_RGB2GRAY);//灰度图
    medianBlur(gary, gary, 3);//中值滤波
    threshold(gary, thresh, 0, 255, cv::THRESH_OTSU);//二值化
    srcImg1 = thresh.clone();
    dstImg = cv::Mat(srcImg.size(), CV_8UC3, cv::Scalar(255, 255, 255));
    //寻找轮廓
    std::vector<std::vector<cv::Point>>contours;
    findContours(srcImg1, contours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
    std::vector<cv::Point>approx;


std::vector<cv::Point>sanjiao;


//画三角形
    for (size_t i = 0; i < contours.size(); i++)
    {
        approxPolyDP(contours[i], approx, arcLength(cv::Mat(contours[i]), true)*0.1, true);
        if (approx.size() == 3 && fabs(contourArea(cv::Mat(approx))) > 1000 && isContourConvex(cv::Mat(approx)))//3条边；面积大于1000；凸多边形
        {
            double minDist = 1e10;

            for (int i = 0; i < 3; i++)//求最小边
            {
                cv::Point side = approx[i] - approx[(i + 1) %3];
                double squaredSideLength = side.dot(side);
                minDist = std::min(minDist, squaredSideLength);
            }
            if (minDist<1000)
                continue;
            for (int i = 0; i<3; i++)
                sanjiao.push_back(cv::Point(approx[i].x, approx[i].y));
            drawContours(dstImg, contours, i, cv::Scalar(255, 0, 0), 3);
        }
    }
    for (size_t i = 0; i < sanjiao.size(); i += 3)
    {
        cv::Point center;
        center.x = (sanjiao[i].x + sanjiao[i + 1].x + sanjiao[i + 2].x) / 3;
        center.y = (sanjiao[i].y + sanjiao[i + 1].y + sanjiao[i + 2].y) / 3;
        std::cout << "三角形中心" << (i + 1) % 3 << center << std::endl;
        circle(dstImg, center, 3, cv::Scalar(255, 0, 0), -1);
//	cvInitFont(&font,CV_FONT_VECTOR0,1,1,0,1,8); fail
       // cvPutText(dstImg,center,cvPoint(center.x+1,center.y),&font,CV_RGB(200,255,255));//show the coordinates of the central point
//      for (;;) {
//     if(!WriteFile(hcom,data,8,&dwWrittenLen,NULL))
//    {   
//         fprintf(stderr, "发送数据失败！\n");
//     }
//                 
//                 k++;
//  std::cout << "三角形中心" << (i + 1) % 3 << center << std::endl;//send data
//                  
//                  printf("往串口发送数据成功！第%d次\n" ,k);
           
//       }

   
   
    }



		namedWindow("Control", CV_WINDOW_AUTOSIZE);
		cvCreateTrackbar("redThre", "Control", &redThre, 255); 
		cvCreateTrackbar("saturationTh", "Control", &saturationTh, 255); 
		CheckColor(frame);
                imshow("1",dstImg);	
                  

                  c = cvWaitKey(30);
                  if (c == 27)//Esc键退出
                     {
                          break;
                      }   



	}	
	return 0;
}
 
//The Color Check is According to "An Early Fire-Detection Method Based on Image Processing"  
 
Mat CheckColor(Mat &inImg)
{
	Mat fireImg;
	fireImg.create(inImg.size(), CV_8UC1);	
	Mat multiRGB[3];
	int a = inImg.channels();
	split(inImg, multiRGB); //将图片拆分成R,G,B,三通道的颜色  
 
	for (int i = 0; i < inImg.rows; i++)
	{
		for (int j = 0; j < inImg.cols; j++)
		{
			float B, G, R;
			B = multiRGB[0].at<uchar>(i, j); //每个像素的R,G,B值,动态地址计算法  
			G = multiRGB[1].at<uchar>(i, j);
			R = multiRGB[2].at<uchar>(i, j);
 
			float maxValue = max(max(B, G), R);
			float minValue = min(min(B, G), R);
			//与HSI中S分量的计算公式
			double S = (1 - 3.0*minValue / (R + G + B));//
			
			//R > RT  R>=G>=B  S>=((255-R)*ST/RT)  
			if (R > redThre &&R >= G && G>= B && S >((255 - R) * saturationTh / redThre))
			{
				fireImg.at<uchar>(i, j) = 255;
			}
			else
			{
				fireImg.at<uchar>(i, j) = 0;
			}
		}
	}
 
	//erode(fireImg, fireImg, Mat(3, 3, CV_8UC1));
	//GaussianBlur(fireImg, fireImg, Size(5, 5), 0, 0);
	medianBlur(fireImg, fireImg, 5);
	dilate(fireImg, fireImg, Mat(5, 5, CV_8UC1));		
    imshow("Binary", fireImg);
	DrawFire(inImg, fireImg);
	return fireImg;
}
 
void DrawFire(Mat &inputImg, Mat foreImg)
{
	vector<vector<Point>> contours_set;//保存轮廓提取后的点集及拓扑关系  
	findContours(foreImg, contours_set, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);	
	Point point1;
	Point point2;	
	float a = 0.4, b = 0.75;
	float xmin1 = a*inputImg.cols, ymin1 = inputImg.rows, xmax1 = 0, ymax1 = 0;
	float xmin2 = b*inputImg.cols, ymin2 = inputImg.rows, xmax2 = a*inputImg.cols, ymax2 = 0;
	float xmin3 = inputImg.cols, ymin3 = inputImg.rows, xmax3 = b*inputImg.cols, ymax3 = 0;
	Rect finalRect1;
	Rect finalRect2;
	Rect finalRect3;	
	vector<vector<Point> >::iterator iter = contours_set.begin();
	for (; iter != contours_set.end();)
	{
		Rect rect = boundingRect(*iter);
		float radius;
		Point2f center;
		minEnclosingCircle(*iter, center, radius);
 
		if (rect.area()> 0)
		{			
			point1.x = rect.x;
			point1.y = rect.y;
			point2.x = point1.x + rect.width;
			point2.y = point1.y + rect.height;
		
			if (point2.x< a*inputImg.cols)
			{
				if (point1.x < xmin1)				
					xmin1 = point1.x;
				if (point1.y < ymin1)
					ymin1 = point1.y;				
				if (point2.x > xmax1 && point2.x < xmax2)				
					xmax1 = point2.x;
				if (point2.y > ymax1)
					ymax1 = point2.y;				
			}
 
			if (point2.x < b*inputImg.cols&&point2.x > a*inputImg.cols)
			{
				if (point1.x < xmin2 && point1.x>xmin1)				
					xmin2 = point1.x;
				if (point1.y < ymin2)
					ymin2 = point1.y;
				if (point2.x > xmax2 && point2.x < xmax3)				
					xmax2 = point2.x;
				if (point2.y > ymax2)
					ymax2 = point2.y;				
			}
 
			if (point2.x < inputImg.cols&&point2.x > b*inputImg.cols)
			{
				if (point1.x < xmin3 && point1.x>xmin2)				
					xmin3 = point1.x;
				if (point1.y < ymin3)
					ymin3 = point1.y;				
				if (point2.x > xmax3)				
					xmax3 = point2.x;
				if (point2.y > ymax3)
					ymax3 = point2.y;				
			}
			
			++iter;
		}
		else
		{
			iter = contours_set.erase(iter);
		}
		
	}
 
	
	if (xmin1 == a*inputImg.cols&& ymin1 == inputImg.rows&&xmax1 == 0 && ymax1== 0)
	{
		xmin1 = ymin1 = xmax1 = ymax1 = 0;
	}
	if (xmin2 == b*inputImg.cols&& ymin2 == inputImg.rows&& xmax2 == a*inputImg.cols&& ymax2 == 0)
	{
		xmin2 = ymin2 = xmax2 = ymax2 = 0;
	}
	if (xmin3 == inputImg.cols&&ymin3 == inputImg.rows&& xmax3 == b*inputImg.cols&& ymax3 == 0)
	{
		xmin3 = ymin3 = xmax3 = ymax3 = 0;
	}
	finalRect1= Rect(xmin1, ymin1, xmax1 - xmin1, ymax1 - ymin1);
	finalRect2 = Rect(xmin2, ymin2, xmax2 - xmin2, ymax2 - ymin2);
	finalRect3 = Rect(xmin3, ymin3, xmax3 - xmin3, ymax3 - ymin3);
	rectangle(inputImg, finalRect1, Scalar(0, 255, 0));
	rectangle(inputImg, finalRect2, Scalar(0, 255, 0));
	rectangle(inputImg, finalRect3, Scalar(0, 255, 0));
	imshow("Fire_Detection", inputImg);	
}
 
