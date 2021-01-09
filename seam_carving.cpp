#include <iostream>
#include <cmath>
#include <sstream>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

string GetString(const string message);
schar WhichMin(const uchar a, const uchar b);
schar WhichMin(const uchar a, const uchar b, const uchar c);

void RGB2Gray(const Mat src, Mat& dst);
void SobelOperation(const Mat src, Mat& dst);
void FindASeamVertical(const Mat src, int* seam);
void FindASeamHorizontal(const Mat src, int* seam);
void RemoveASeamVertical(const Mat src, Mat& dst, const int* seam);
void RemoveASeamHorizontal(const Mat src, Mat& dst, const int* seam);

void SeamCarvingVertical(const Mat src, Mat& dst, int seamNum);
void SeamCarvingHorizontal(const Mat src, Mat& dst, int seamNum);

//Seam Carving Loop
//有一張原始彩色圖(m*n)
//使用其彩色圖(m*n)計算灰階圖(m*n)
//使用灰階圖(m*n)計算Sobel Edge Detection(m*n)
//使用sobel(m*n), 彩色圖(m*n) 進行seam carving 可得一彩色圖((m-1)*n) or (m*(n-1))
//
//再使用此彩色圖
//計算其灰階圖
//......

int main(int argc, char** argv)
{
	//get image name
	string fileIn = GetString("Please Enter the Image File Name: ");

	Mat src, dst;
	
	//read image
	src = imread(fileIn, CV_LOAD_IMAGE_COLOR);
	if(src.empty())
	{
		cout << "Can't Find File!" << endl;
		system("pause");
		return -1;
	}

	//show original image detail
	cout << endl;
	cout << "========== Source Image ==========" << endl;
	cout << " * File Name: " << fileIn << endl;
	cout << " * Size: " << src.cols << " * " << src.rows << endl;
	cout << "==================================" << endl << endl;

	//show original image
	namedWindow(fileIn, CV_WINDOW_AUTOSIZE);
	imshow(fileIn, src);
	waitKey();
	destroyWindow(fileIn);
	
	//get scale direction
	int dir;
	cout << "Please Choose the Scale Direction: " << endl;
	cout << " * 1. Vertical" << endl;
	cout << " * 2. Horizontal" << endl;
	cin >> dir;
	if(dir != 1 && dir != 2)
	{
		cout << "Wrong Input. Please Enter the Number '1' or '2'." << endl;
		system("pause");
		return -1;
	}

	//get seam amount
	int seamNum;
	cout << "Please Enter the Pixel Amount You Want to Delete: ";
	cin >> seamNum;
	if( (dir == 1 && seamNum > src.rows - 4) || (dir == 2 && seamNum > src.cols - 4) )
	{
		cout << "Wrong Input. The Pixel Amount is too Big." << endl;
		system("pause");
		return -1;
	}

	//set output image name
	stringstream ss;
	ss << "_seamc_";
	if(dir == 1)
		ss << "v";
	else
		ss << "h";
	ss << seamNum;

	string fileOut = fileIn;
	fileOut.insert(fileOut.find('.'), ss.str());

	//do seam carving
	if(dir == 1)
	{
		cout << "Vertical Scale. Remove " << seamNum << " Pixels." << endl;
		SeamCarvingVertical(src, dst, seamNum);
	}
	if(dir == 2)
	{
		cout << "Horizontal Scale. Remove " << seamNum << " Pixels." << endl;
		SeamCarvingHorizontal(src, dst, seamNum);
	}

	//show result image detail
	cout << endl;
	cout << "========== Result Image ==========" << endl;
	cout << " * File Name: " << fileOut << endl;
	cout << " * Size: " << dst.cols << " * " << dst.rows << endl;
	cout << "==================================" << endl << endl;
	
	//save result image
	imwrite(fileOut, dst);

	//show result image
	namedWindow(fileOut, CV_WINDOW_AUTOSIZE);
	imshow(fileOut, dst);
	waitKey();
	destroyWindow(fileOut);

    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

string GetString(const string message)
{
	char* input = new char[256];

	cout << message;

	cin.getline(input, 256, '\n');
	string fileIn = input;

	delete [] input;

	return fileIn;
}

schar WhichMin(const uchar a, const uchar b)
{
	if(min(a, b) == a)
		return 0;
	else
		return 1;
}

schar WhichMin(const uchar a, const uchar b, const uchar c)
{
	if(min(a, min(b, c)) == a)
		return -1;
	else if(min(a, min(b, c)) == b)
		return 0;
	else
		return 1;
}

void RGB2Gray(const Mat src, Mat& dst)
{
	if(!dst.empty())
		dst.release();

	dst = Mat::zeros(src.size(), CV_8UC1);

	for(int y = 0; y < src.rows; ++y)
	{
		for(int x = 0; x < src.cols; ++x)
		{
			//gray value = (r + g + b) / 3
			dst.at<uchar>(Point(x, y)) = (src.at<Vec3b>(Point(x, y))[0] + src.at<Vec3b>(Point(x, y))[1] + src.at<Vec3b>(Point(x, y))[2])/3;
		}
	}
}

void SobelOperation(const Mat src, Mat& dst)
{
	int **gray;
	int **gradientMap;
	int sobel_filter_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
	int sobel_filter_y[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
	int gmin, gmax;
	float gx = 0, gy = 0;
	
	//memory allocation
	gray = new int*[src.cols];
	for(int i = 0; i < src.cols; ++i)
		gray[i] = new int[src.rows];
	gradientMap = new int*[src.cols];
	for(int i = 0; i < src.cols; ++i)
		gradientMap[i] = new int[src.rows];

	//copy image data
	for(int y = 0; y < src.rows; ++y)
	{
		for(int x = 0; x < src.cols; ++x)
		{
			gray[x][y] = src.at<uchar>(Point(x, y));
		}
	}

	//use sobel filter to compute gradient map
	for(int y = 1; y < src.rows - 1; ++y)
	{
		for(int x = 1; x < src.cols - 1; ++x)
		{
			gx = 0;
			gy = 0;
			
			//do sobel filter
			for(int n = y - 1; n < y + 2; ++n)
			{
				for(int m = x - 1; m < x + 2; ++m)
				{
					gx += gray[m][n] * sobel_filter_x[m - x + 1][n - y + 1];
					gy += gray[m][n] * sobel_filter_y[m - x + 1][n - y + 1];
				}
			}
			
			gradientMap[x][y] = (int)sqrt(gx * gx + gy * gy);

			//find the maximum and minimum gradient
			if(x == 1 && y == 1)
			{
				gmin = gmax = gradientMap[1][1];
			}
			else
			{
				int value = gradientMap[x][y];
				if(gmin > value)
					gmin = value;
				else if(gmax < value)
					gmax = value;
			}
		}
	}
	
	//change the scale from minimum - maximum to 0 - 255
	//0       - 255			scaleA
	//minimum - maximum		scaleB
	//scaleA = 255 * (scaleB - minimum) / (maximum - minimum)
	if(!dst.empty())
		dst.release();

	dst = Mat::zeros(src.size(), CV_8UC1);
	int diff = gmax - gmin;
	for(int y = 1; y < dst.rows - 1; ++y)
	{
		for(int x = 1; x < dst.cols - 1; ++x)
		{
			dst.at<uchar>(Point(x, y)) = 255 * (gradientMap[x][y] - gmin) / diff;
		}
	}

	for(int i = 0; i < src.cols; ++i)
		delete [] gray[i];
	delete [] gray;

	for(int i = 0; i < src.cols; ++i)
		delete [] gradientMap[i];
	delete [] gradientMap;
}

void FindASeamVertical(const Mat src, int* seam)
{
	int currentValue = 0;
	int minimumValue = 0;
	int* currentSeam = new int[src.cols];
	int* minimumSeam = new int[src.cols];

	for(int y = 1; y < src.rows - 1; ++y)
	{
		int cy;

		cy = currentSeam[1] = y;
		currentValue = 0;
		currentValue += src.at<uchar>(Point(1, cy));

		for(int x = 2; x < src.cols - 1; ++x)
		{
			//edge point
			if(cy == 1)
			{
				int tmp = WhichMin(src.at<uchar>(Point(x, cy)), src.at<uchar>(Point(x, cy + 1)));
				currentValue += src.at<uchar>(Point(x, cy + tmp));
				cy = currentSeam[x] = cy + tmp;
			}
			//edge point
			else if(cy == src.rows - 2)
			{
				int tmp = WhichMin(src.at<uchar>(Point(x, cy - 1)), src.at<uchar>(Point(x, cy)));
				currentValue += src.at<uchar>(Point(x, cy + tmp - 1));
				cy = currentSeam[x] = cy + tmp - 1;
			}
			else
			{
				int tmp = WhichMin(src.at<uchar>(Point(x, cy - 1)), src.at<uchar>(Point(x, cy)), src.at<uchar>(Point(x, cy + 1)));
				currentValue += src.at<uchar>(Point(x, cy + tmp));
				cy = currentSeam[x] = cy + tmp;
			}
		}

		if(y == 1)
		{
			minimumValue = currentValue;
			minimumSeam = currentSeam;
			currentSeam = new int[src.cols];
		}
		
		if(currentValue < minimumValue)
		{
			delete [] minimumSeam;
			minimumValue = currentValue;
			minimumSeam = currentSeam;
			currentSeam = new int[src.cols];
		}
	}

	minimumSeam[0] = minimumSeam[1];
	minimumSeam[src.cols - 1] = minimumSeam[src.cols - 2];
	for(int i = 0; i < src.cols; ++i)
		seam[i] = minimumSeam[i];

	delete [] currentSeam;
	delete [] minimumSeam;
}

void FindASeamHorizontal(const Mat src, int* seam)
{
	int currentValue = 0;
	int minimumValue = 0;
	int* currentSeam = new int[src.rows];
	int* minimumSeam = new int[src.rows];

	for(int x = 1; x < src.cols - 1; ++x)
	{
		int cx;

		cx = currentSeam[1] = x;
		currentValue = 0;
		currentValue += src.at<uchar>(Point(cx, 1));

		for(int y = 2; y < src.rows - 1; ++y)
		{
			//edge point
			if(cx == 1)
			{
				int tmp = WhichMin(src.at<uchar>(Point(cx, y)), src.at<uchar>(Point(cx + 1, y)));
				currentValue += src.at<uchar>(Point(cx + tmp, y));
				cx = currentSeam[y] = cx + tmp;
			}
			//edge point
			else if(cx == src.cols - 2)
			{
				int tmp = WhichMin(src.at<uchar>(Point(cx - 1, y)), src.at<uchar>(Point(cx, y)));
				currentValue += src.at<uchar>(Point(cx + tmp - 1, y));
				cx = currentSeam[y] = cx + tmp - 1;
			}
			else
			{
				int tmp = WhichMin(src.at<uchar>(Point(cx - 1, y)), src.at<uchar>(Point(cx, y)), src.at<uchar>(Point(cx + 1, y)));
				currentValue += src.at<uchar>(Point(cx + tmp, y));
				cx = currentSeam[y] = cx + tmp;
			}
		}

		if(x == 1)
		{
			minimumValue = currentValue;
			minimumSeam = currentSeam;
			currentSeam = new int[src.rows];
		}
		
		if(currentValue < minimumValue)
		{
			delete minimumSeam;
			minimumValue = currentValue;
			minimumSeam = currentSeam;
			currentSeam = new int[src.rows];
		}
	}

	minimumSeam[0] = minimumSeam[1];
	minimumSeam[src.rows - 1] = minimumSeam[src.rows - 2];
	for(int i = 0; i < src.rows; ++i)
		seam[i] = minimumSeam[i];

	delete [] currentSeam;
	delete [] minimumSeam;
}

void RemoveASeamVertical(const Mat src, Mat& dst, const int* seam)
{
	dst = Mat::zeros(src.rows - 1, src.cols, CV_8UC3);

	
	for(int x = 0; x < dst.cols; ++x)
	{
		for(int y = 0, ty = 0; y < dst.rows; ++y)
		{
			//skip the point on seam
			if(ty == seam[x])
				ty++;

			dst.at<Vec3b>(Point(x, y))[0] = src.at<Vec3b>(Point(x, ty))[0];
			dst.at<Vec3b>(Point(x, y))[1] = src.at<Vec3b>(Point(x, ty))[1];
			dst.at<Vec3b>(Point(x, y))[2] = src.at<Vec3b>(Point(x, ty))[2];

			ty++;
		}
	}
}

void RemoveASeamHorizontal(const Mat src, Mat& dst, const int* seam)
{
	dst = Mat::zeros(src.rows, src.cols - 1, CV_8UC3);

	for(int y = 0; y < dst.rows; ++y)
	{
		for(int x = 0, tx = 0; x < dst.cols; ++x)
		{
			//skip the point on seam
			if(tx == seam[y])
				tx++;

			dst.at<Vec3b>(Point(x, y))[0] = src.at<Vec3b>(Point(tx, y))[0];
			dst.at<Vec3b>(Point(x, y))[1] = src.at<Vec3b>(Point(tx, y))[1];
			dst.at<Vec3b>(Point(x, y))[2] = src.at<Vec3b>(Point(tx, y))[2];

			tx++;
		}
	}
}

void SeamCarvingVertical(const Mat src, Mat& dst, int seamNum)
{
	Mat gray, gradient;

	dst = src;

	for(int i = 0; i < seamNum; ++i)
	{
		int *seam = new int[dst.cols];

		RGB2Gray(dst, gray);
		SobelOperation(gray, gradient);
		FindASeamVertical(gradient, seam);
		RemoveASeamVertical(dst, dst, seam);

		cout << "\r";
		cout << "Removed " << i + 1 << " Seams...";
		delete [] seam;
	}

	cout << endl << "Seam Carving Done." << endl;
}

void SeamCarvingHorizontal(const Mat src, Mat& dst, int seamNum)
{
	Mat gray, gradient;

	dst = src;

	for(int i = 0; i < seamNum; ++i)
	{
		int *seam = new int[dst.rows];

		RGB2Gray(dst, gray);
		SobelOperation(gray, gradient);
		FindASeamHorizontal(gradient, seam);
		RemoveASeamHorizontal(dst, dst, seam);

		cout << "\r";
		cout << "Removed " << i + 1 << " Seams...";
		delete [] seam;
	}

	cout << endl << "Seam Carving Done." << endl;
}