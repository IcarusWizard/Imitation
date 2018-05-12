
// ImitationDlg.cpp : 实现文件
//

#include "stdafx.h"
#include "Imitation.h"
#include "ImitationDlg.h"
#include "afxdialogex.h"
#include <opencv.hpp>
#include <Kinect.h>
#include <thread>
#include <time.h>
#include "Motor.h"
#include "myVector.h"
#include <io.h> 
#include <fcntl.h>
#include <conio.h>
#include <vector>
#include <fstream>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

using namespace std;
using namespace cv;

#pragma region Global Variables
const double pi = 3.1415926;
double theta[4];
int controlTheta[7];
/*Define Global Variables*/
bool Run;
IKinectSensor *myKinectSensor;
ICoordinateMapper *myCoordinateMapper;
//Color Frame Variables
IColorFrameSource *myColorFrameSource;
IColorFrameReader *myColorFrameReader;
IColorFrame *myColorFrame;
Mat img_rgb(1080, 1920, CV_8UC4);
Mat rgb_save(1080, 1920, CV_8UC4);
//Depth Frame Variables
IDepthFrameSource *myDepthFrameSource;
IDepthFrameReader *myDepthFrameReader;
IDepthFrame *myDepthFrame;
Mat Depth_data(424, 512, CV_16UC1);
Mat img_depth(424, 512, CV_8UC1);
Mat depth_save(424, 512, CV_8UC1);
//Body Frame Variables
IBodyFrameSource *myBodyFrameSource;
IBodyFrameReader *myBodyFrameReader;
IBodyFrame *myBodyFrame;
CameraSpacePoint points[JointType_Count];
ColorSpacePoint Cpoints[JointType_Count];
DepthSpacePoint Dpoints[JointType_Count];
Point Cdraw[JointType_Count];
Point Ddraw[JointType_Count];
bool Track; // Not using
Motor *myMotor;
bool depthSaved = false;
bool colorSaved = false;
int startTime;
bool enableMotor = false;
bool openedMotor = false;
thread *ColorThread;
thread *DepthThread;
thread *BodyThread;
vector<double*> thetaRobot;
vector<double*> thetaProcess;
vector<Vector3D*> elbowRaw;
vector<Vector3D*> wristRaw;
Vector3D elbowTotal(0);
Vector3D wristTotal(0);
int outputPeriod = 4;
int filtersize = 100;
#pragma endregion
#pragma region Global Function
// Define Realease Function
template<class Interface>
inline void SafeRelease(Interface *& pInterfaceToRelease)
{
	if (pInterfaceToRelease != NULL)
	{
		pInterfaceToRelease->Release();
		pInterfaceToRelease = NULL;
	}
}
//Inverse Kinematics
void InverseKinematics(double *theta, Vector3D shoulder, Vector3D elbow, Vector3D wrist)
{
	Vector3D Vec1 = wrist - elbow;
	norm(Vec1);
	norm(elbow);

	theta[1] = acos(elbow.z);
	if (elbow.y < 0)
		theta[1] = -theta[1];

	theta[0] = acos(elbow.x / sin(theta[1]));

	theta[3] = angle(elbow, Vec1) - pi / 2;
	theta[3] = -theta[3];

	Vector3D nVec = cross(elbow, Vec1);
	norm(nVec);
	theta[2] = asin(nVec.z / sin(theta[1]));
}
// Delay Function
void Delay(int ms)
{
	int endtime = ms + clock();
	while (clock() < endtime);
}
// Function to Draw lines
inline void DrawHelper(Mat &img, Point *&DrawPoints, const int *Index, int length)
{
	Point point2draw[10] = {};
	for (int i = 0; i < length; ++i)
		point2draw[i] = DrawPoints[Index[i]];
	for (int i = 0; i < length - 1; ++i)
		line(img, point2draw[i], point2draw[i + 1], Scalar(0, 255, 0), 2);
}
// Function to Draw Skeleton
inline void DrawSkeleton(Mat &img, Point DrawPoints[25], int r = 20, const int length = JointType_Count)  // inline here may improve the speed
{
	//Index of the Skeleton data
	const int LeftArm[] = { 20, 4, 5, 6, 7, 21 };
	const int RightArm[] = { 20, 8, 9, 10, 11, 23 };
	const int LeftThumb[] = { 6, 22 };
	const int RightThumb[] = { 10, 24 };
	const int LeftLeg[] = { 0, 12, 13, 14, 15 };
	const int RightLeg[] = { 0, 16, 17, 18, 19 };
	const int Spinal[] = { 3, 2, 20, 1, 0 };
	//--------------------------------------------
	for (int i = 0; i < length; ++i)
		circle(img, DrawPoints[i], r, Scalar(0, 255, 0), -1);
	DrawHelper(img, DrawPoints, LeftArm, 6);
	DrawHelper(img, DrawPoints, RightArm, 6);
	DrawHelper(img, DrawPoints, LeftThumb, 2);
	DrawHelper(img, DrawPoints, RightThumb, 2);
	DrawHelper(img, DrawPoints, LeftLeg, 5);
	DrawHelper(img, DrawPoints, RightLeg, 5);
	DrawHelper(img, DrawPoints, Spinal, 5);
}
// Process Arm state
void processArmState(CameraSpacePoint points[25])
{
	Vector3D left_shoulder(points[4]);
	Vector3D left_elbow(points[5]);
	Vector3D left_wrist(points[6]);
	Vector3D left_hand(points[7]);

	Vector3D right_shoulder(points[8]);
	Vector3D right_elbow(points[9]);
	Vector3D right_wrist(points[10]);
	Vector3D right_hand(points[11]);

	left_shoulder = leftTrans(left_shoulder);
	left_elbow = leftTrans(left_elbow) - left_shoulder;
	left_wrist = leftTrans(left_wrist) - left_shoulder;
	left_hand = leftTrans(left_hand) - left_shoulder;
	left_shoulder = left_shoulder - left_shoulder;

	right_shoulder = rightTrans(right_shoulder);
	right_elbow = rightTrans(right_elbow) - right_shoulder;
	right_wrist = rightTrans(right_wrist) - right_shoulder;
	right_hand = rightTrans(right_hand) - right_shoulder;
	right_shoulder = right_shoulder - right_shoulder;

	int cnt = elbowRaw.size();
	if (cnt < filtersize)
	{
		elbowTotal = elbowTotal + right_elbow;
		wristTotal = wristTotal + right_wrist;
	}
	else
	{
		elbowTotal = elbowTotal + right_elbow - *elbowRaw[cnt - filtersize];
		wristTotal = wristTotal + right_wrist - *wristRaw[cnt - filtersize];
	}
	Vector3D* tem = new Vector3D(right_elbow);
	elbowRaw.push_back(tem);
	tem = new Vector3D(right_wrist);
	wristRaw.push_back(tem);
	InverseKinematics(theta, right_shoulder, elbowTotal / filtersize, wristTotal / filtersize);
	//_cprintf("Inverse Kinematics Result:theta1: %.2f, theta2: %.2f, theta3: %.2f, theta4: %.2f\n", theta[0], theta[1], theta[2], theta[3]);
}
void InitConsoleWindow()
{
	AllocConsole();
	HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);
	int hCrt = _open_osfhandle((long)handle, _O_TEXT);
	FILE * hf = _fdopen(hCrt, "w");
	*stdout = *hf;
}
#pragma endregion
//-----------------------------------------
#pragma region Color
//A Clock Function to Find The fps of Color Frame
void countClockColor()
{
	static int time = 0;
	int nowtime = clock();
	fprintf(stdout, "The Color Frame Cost time %d ms\n", nowtime - time);
	time = nowtime;
}
//Color Frame Function
void ColorFrame()
{
	const UINT buffer_size = 1920 * 1080 * 4;
	const int r = 20;
	namedWindow("ColorFrame", 0);
	while (Run && (!depthSaved || enableMotor))
	{
		if (myColorFrameReader->AcquireLatestFrame(&myColorFrame) == S_OK &&
			myColorFrame->CopyConvertedFrameDataToArray(buffer_size, reinterpret_cast<BYTE*>(img_rgb.data), ColorImageFormat::ColorImageFormat_Bgra) == S_OK)
		{
			//countClockColor();
			//DrawSkeleton(img_rgb, Cdraw);
			imshow("ColorFrame", img_rgb);
			if (clock() - startTime > 5000 && !colorSaved)
			{
				//rgb_save = img_rgb;
				//colorSaved = true;
			}
			SafeRelease(myColorFrame);
			if (waitKey(1) == VK_ESCAPE)
				break;
		}
	}
	Run = false;
}
#pragma endregion
#pragma region Depth
//A Clock Function to Find The fps of Depth Frame
void countClockDepth()
{
	static int time = 0;
	int nowtime = clock();
	fprintf(stdout, "The Depth Frame Cost time %d ms\n", nowtime - time);
	time = nowtime;
}
//Depth Frame Function
void DepthFrame()
{
	const UINT buffer_size = 424 * 512;
	const int r = 20;
	//namedWindow("DepthFrame");
	while (Run && (!colorSaved || enableMotor))
	{
		if (myDepthFrameReader->AcquireLatestFrame(&myDepthFrame) == S_OK &&
			myDepthFrame->CopyFrameDataToArray(buffer_size, reinterpret_cast<UINT16*>(Depth_data.data)) == S_OK)
		{
			//countClockDepth();
			for (int i = 0; i < 424; ++i)
				for (int j = 0; j < 512; ++j)
					img_depth.at<BYTE>(i, j) = Depth_data.at<UINT16>(i, j) % 256;
			DrawSkeleton(img_depth, Ddraw, 10);
			imshow("DepthFrame", img_depth);
			if (clock() - startTime > 5000 && !depthSaved)
			{
				//depth_save = img_depth;
				//depthSaved = true;
			}
			SafeRelease(myDepthFrame);
			if (waitKey(1) == VK_ESCAPE)
				break;
		}
	}
	Run = false;
}
#pragma endregion
#pragma region Body
//A Clock Function to Find The fps of Body Frame
void countClockBody()
{
	static int time = 0;
	int nowtime = clock();
	fprintf(stdout, "The Body Frame Cost time %d ms\n", nowtime - time);
	time = nowtime;
}
//Body Frame Function
void BodyFrame()
{
	IBody* Bodys[BODY_COUNT] = { 0 };
	while (Run)
	{
		if (myBodyFrameReader->AcquireLatestFrame(&myBodyFrame) == S_OK)
		{
			myBodyFrame->GetAndRefreshBodyData(_countof(Bodys), Bodys);
			Track = false;
			for (int i = 0; i < BODY_COUNT; ++i)
			{
				IBody *pBody = Bodys[i];
				if (pBody)
				{
					BOOLEAN isTracked = false;
					if (pBody->get_IsTracked(&isTracked) == S_OK && isTracked)
					{
						Joint joints[JointType_Count];
						if (pBody->GetJoints(JointType_Count, joints) == S_OK)
						{
							for (int j = 0; j < JointType_Count; ++j)
							{
								points[j] = joints[j].Position;
								//fprintf(stdout, "%f\t%f\t%f\n", points[j].X, points[j].Y, points[j].Z); // cost too much time!
							}
						}
						if (enableMotor)
						{
							processArmState(points);
							double* tem = new double[4];
							for (auto i = 0; i < 4; ++i)
								tem[i] = theta[i];
							thetaProcess.push_back(tem);
							double *rtheta = new double[4];
							myMotor->getTheta(rtheta);
							thetaRobot.push_back(rtheta);
							int cnt = elbowRaw.size();
							if (cnt % outputPeriod == 0 && cnt > filtersize)
							{
								_cprintf("Inverse Kinematics Result:theta1: %.2f, theta2: %.2f, theta3: %.2f, theta4: %.2f\n", theta[0], theta[1], theta[2], theta[3]);
								myMotor->Move(Left, theta);
							}
							
						}
						Track = true;
						myCoordinateMapper->MapCameraPointsToColorSpace(JointType_Count, points, JointType_Count, Cpoints);
						myCoordinateMapper->MapCameraPointsToDepthSpace(JointType_Count, points, JointType_Count, Dpoints);
						for (int j = 0; j < JointType_Count; ++j)
						{
							Cdraw[j].x = int(Cpoints[j].X);
							Cdraw[j].y = int(Cpoints[j].Y);
							Ddraw[j].x = int(Dpoints[j].X);
							Ddraw[j].y = int(Dpoints[j].Y);
						}
					}
				}
			}
			countClockBody();
			SafeRelease(myBodyFrame);
		}
	}
	Run = false;
}
#pragma endregion

// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CImitationDlg 对话框



CImitationDlg::CImitationDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_IMITATION_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CImitationDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CImitationDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDOK, &CImitationDlg::OnBnClickedOk)
	ON_BN_CLICKED(openKinect, &CImitationDlg::OnBnClickedopenkinect)
	ON_BN_CLICKED(openMotor, &CImitationDlg::OnBnClickedopenmotor)
	ON_BN_CLICKED(releaseKinect, &CImitationDlg::OnBnClickedreleasekinect)
	ON_BN_CLICKED(startTracking, &CImitationDlg::OnBnClickedstarttracking)
	ON_BN_CLICKED(closeMotor, &CImitationDlg::OnBnClickedclosemotor)
	ON_BN_CLICKED(stopTracking, &CImitationDlg::OnBnClickedstoptracking)
	ON_BN_CLICKED(startImitation, &CImitationDlg::OnBnClickedstartimitation)
	ON_BN_CLICKED(stopImitation, &CImitationDlg::OnBnClickedstopimitation)
	ON_BN_CLICKED(motorHome, &CImitationDlg::OnBnClickedmotorhome)
	ON_BN_CLICKED(Photo, &CImitationDlg::OnBnClickedPhoto)
END_MESSAGE_MAP()


// CImitationDlg 消息处理程序

BOOL CImitationDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();
	elbowTotal.x = 0;
	elbowTotal.y = 0;
	elbowTotal.z = 0;
	wristTotal.x = 0;
	wristTotal.y = 0;
	wristTotal.z = 0;
	InitConsoleWindow();
	_cprintf("Open console OK\n\n");
	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CImitationDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CImitationDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CImitationDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CImitationDlg::OnBnClickedOk()
{
	// TODO: 在此添加控件通知处理程序代码
	CDialogEx::OnOK();
}

void CImitationDlg::OnBnClickedopenkinect()
{
	// TODO: 在此添加控件通知处理程序代码
#pragma region Open Sensor
	HRESULT hr;
	// Open Kinect Sensor
	hr = GetDefaultKinectSensor(&myKinectSensor);
	if (SUCCEEDED(hr))
		hr = myKinectSensor->Open();
	if (FAILED(hr)) {
		_cprintf( "Open Kinect Sensor Error!\n");
		return;
	}
	_cprintf( "Kinect Sensor Ready!\n");
	// Get CoordinateMapper
	hr = myKinectSensor->get_CoordinateMapper(&myCoordinateMapper);
	if (FAILED(hr)) {
		_cprintf("Get Coordinate Mapper Error\n");
		return;
	}
	_cprintf("Coordinate Mapper Ready!\n");
	// Get Color Frame Source
	hr = myKinectSensor->get_ColorFrameSource(&myColorFrameSource);
	if (FAILED(hr)) {
		_cprintf("Get Color Frame Source Error\n");
		return;
	}
	_cprintf("Color Frame Source Ready!\n");
	// Open Color Frame Reader
	hr = myColorFrameSource->OpenReader(&myColorFrameReader);
	if (FAILED(hr)) {
		_cprintf("Open Color Reader Error\n");
		return;
	}
	_cprintf("Color Frame Reader Ready!\n");
	// Get Depth Frame Source
	hr = myKinectSensor->get_DepthFrameSource(&myDepthFrameSource);
	if (FAILED(hr)) {
		_cprintf("Get Depth Frame Source Error\n");
		return;
	}
	_cprintf("Depth Frame Source Ready!\n");
	// Open Depth Frame Reader
	hr = myDepthFrameSource->OpenReader(&myDepthFrameReader);
	if (FAILED(hr)) {
		_cprintf("Open Depth Reader Error\n");
		return;
	}
	_cprintf("Depth Frame Reader Ready!\n");
	// Get Body Frame Source
	hr = myKinectSensor->get_BodyFrameSource(&myBodyFrameSource);
	if (FAILED(hr)) {
		_cprintf("Get Body Frame Source Error\n");
		return;
	}
	_cprintf("Body Frame Source Ready!\n");
	// Open Body Frame Reader
	hr = myBodyFrameSource->OpenReader(&myBodyFrameReader);
	if (FAILED(hr)) {
		_cprintf("Open Body Frame Reader Error\n");
		return;
	}
	_cprintf("Body Frame Reader Ready!\n");
	BOOLEAN isAvailable = false;
	do {
		myKinectSensor->get_IsAvailable(&isAvailable);
	} while (!isAvailable);
	_cprintf("Kinect sensor is ready!\n\n");
#pragma endregion
}


void CImitationDlg::OnBnClickedopenmotor()
{
	// TODO: 在此添加控件通知处理程序代码
	myMotor = new Motor(5);
	myMotor->servoOpen();
	openedMotor = true;
	_cprintf("Motor Opend!\n\n");
}


void CImitationDlg::OnBnClickedreleasekinect()
{
	// TODO: 在此添加控件通知处理程序代码
	SafeRelease(myColorFrameReader);
	SafeRelease(myColorFrameSource);
	SafeRelease(myDepthFrameReader);
	SafeRelease(myDepthFrameSource);
	SafeRelease(myBodyFrameReader);
	SafeRelease(myBodyFrameSource);
	SafeRelease(myCoordinateMapper);
	myKinectSensor->Close();
	SafeRelease(myKinectSensor);
	_cprintf("Kinect source released!\n\n");
}


void CImitationDlg::OnBnClickedstarttracking()
{
	// TODO: 在此添加控件通知处理程序代码
	Run = true;
	startTime = clock();
	ColorThread = new thread(ColorFrame);
	DepthThread = new thread(DepthFrame);
	BodyThread = new thread(BodyFrame);
	_cprintf("Is Tracking now....\n\n");
}


void CImitationDlg::OnBnClickedclosemotor()
{
	// TODO: 在此添加控件通知处理程序代码
	myMotor->servoClose();
	_cprintf("Motor is closed!\n\n");
}


void CImitationDlg::OnBnClickedstoptracking()
{
	// TODO: 在此添加控件通知处理程序代码
	Run = false;
	_cprintf("Tracking is stopped!\n");
	ofstream f1("thetaRobot.txt", ios::out);
	for (auto i = 0; i < thetaRobot.size(); ++i)
		f1 << thetaRobot[i][0] << " " << thetaRobot[i][1] << " " << thetaRobot[i][2] << " " << thetaRobot[i][3] << endl;
	ofstream f2("thetaProcess.txt", ios::out);
	for (auto i = 0; i < thetaProcess.size(); ++i)
		f2 << thetaProcess[i][0] << " " << thetaProcess[i][1] << " " << thetaProcess[i][2] << " " << thetaProcess[i][3] << endl;
	f2.close();
	ofstream f3("elbowdata.txt", ios::out);
	for (auto i = 0; i < elbowRaw.size(); ++i)
		f3 << *elbowRaw[i];
	f3.flush();
	f3.close();
	ofstream f4("wristdata.txt", ios::out);
	for (auto i = 0; i < wristRaw.size(); ++i)
		f4 << *wristRaw[i];
	f4.flush();
	f4.close();
	_cprintf("File saved!\n\n");
}


void CImitationDlg::OnBnClickedstartimitation()
{
	// TODO: 在此添加控件通知处理程序代码
	enableMotor = true;
	_cprintf("Motor is imitating.....\n\n");
}


void CImitationDlg::OnBnClickedstopimitation()
{
	// TODO: 在此添加控件通知处理程序代码
	enableMotor = false;
	_cprintf("Motor is not imitating any more\n\n");
}


void CImitationDlg::OnBnClickedmotorhome()
{
	// TODO: 在此添加控件通知处理程序代码
	_cprintf("Motor is going to Home position....");
	myMotor->Home(Left);
	_cprintf("Motor is Home now\n\n");
}


void CImitationDlg::OnBnClickedPhoto()
{
	// TODO: 在此添加控件通知处理程序代码
	_cprintf("Smile!");
	int start = clock();
	while (clock() - start < 3000);
	_cprintf("Taking a photo....");
	imwrite("rgb.png", img_rgb);
	imwrite("dep.png", img_depth);
	_cprintf("Photo is OK\n\n");
}
