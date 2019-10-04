// This Program is Written by Abubakr Shafique (abubakr.shafique@gmail.com)
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Inversion_CUDA.h"

using namespace std;
using namespace cv;

int main(){
	Mat Input_Image = imread("Test_Image.png");

	cout << "Height: " << Input_Image.rows << ", Width: " << Input_Image.rows << ", Channels: " << Input_Image.channels() << endl;

	Image_Inversion_CUDA(Input_Image.data, Input_Image.rows, Input_Image.rows, Input_Image.channels());

	imwrite("Inverted_Image.png", Input_Image);
	system("pause");
	return 0;
}