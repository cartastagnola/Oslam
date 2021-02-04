#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/core/mat.hpp>
#include <iostream>

// SDL2
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL2_gfxPrimitives.h>

#include <stdio.h>
#include <string>
#include <typeinfo>

#include "raylib.h"
#define FOVY_PERSPECTIVE    45.0f
#define WIDTH_ORTHOGRAPHIC  10.0f

using namespace cv;
using namespace std;

//Screen dimension constants
const int IMG_WIDTH = 640;
const int IMG_HEIGHT = 480;
const int SCREEN_WIDTH = IMG_WIDTH*3;
const int SCREEN_HEIGHT = IMG_HEIGHT;

enum KeyPressSurfaces
{
	KEY_PRESS_SURFACE_DEFAULT,
	KEY_PRESS_SURFACE_UP,
	KEY_PRESS_SURFACE_DOWN,
	KEY_PRESS_SURFACE_LEFT,
	KEY_PRESS_SURFACE_RIGHT,
	KEY_PRESS_SURFACE_TOTAL
};

//Starts up SDL and creates window
bool init();

//Loads media
bool loadMedia();

//Frees media and shuts down SDL
void close();

//Loads individual image
SDL_Surface* loadSurface( std::string path );

//The window we'll be rendering to
SDL_Window* gWindow = NULL;
	
//The surface contained by the window
SDL_Surface* gScreenSurface = NULL;
//The images that correspond to a keypress
SDL_Surface* gKeyPressSurfaces[ KEY_PRESS_SURFACE_TOTAL ];

//Current displayed image
SDL_Surface* gCurrentSurface = NULL;

//Current rendere
SDL_Renderer* gRenderer = NULL;

//features data
std::vector<KeyPoint> keyPointsCollection[10][200];
cv::Mat descriptorsCollection[10][200];
std::vector<KeyPoint> currentFilteredPointsCollection[10][200];
std::vector<KeyPoint> previousFilteredPointsCollection[10][200];

std::vector<KeyPoint> MkeyPointsCollection[200];
cv::Mat MdescriptorCollection[200];
std::vector<KeyPoint> M2keyPointsCollection[200];
cv::Mat M2descriptorCollection[200];
std::vector<KeyPoint> M3keyPointsCollection[200];
cv::Mat M3descriptorCollection[200];


bool init()
{
	//Initialization flag
	bool success = true;

	//Initialize SDL
	if( SDL_Init( SDL_INIT_VIDEO ) < 0 )
	{
		printf( "SDL could not initialize! SDL Error: %s\n", SDL_GetError() );
		success = false;
	}
	else
	{
		//Create window
		gWindow = SDL_CreateWindow( "cSLAM", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN );
		if( gWindow == NULL )
		{
			printf( "Window could not be created! SDL Error: %s\n", SDL_GetError() );
			success = false;
		}
		else
		{
                    //legacy surface
                    gScreenSurface = SDL_GetWindowSurface( gWindow );

                    //Create renderer for window
                    gRenderer = SDL_CreateRenderer( gWindow, -1, SDL_RENDERER_ACCELERATED );
                    if( gRenderer == NULL )
                    {
			printf( "Window could not be created! SDL Error: %s\n", SDL_GetError() );
			success = false;
                    }
                    else
                    {
                        //Initialized renderer color
                        SDL_SetRenderDrawColor( gRenderer, 0xFF, 0xFF, 0xFF, 0xFF);

                        /*
                        //Initialized PNG loading
                        int imgFlags = IMG_INIT_PNG;
                        if( !( IMG_Init( imgFlags ) & imgFlags ) )
                        {
                            printf( "SDL_image could not initialize! SDL_image Error: %s\n", IMG_GetError() );
                            success = false;
                        }
                        */
                    }

		}

	}

	return success;
}


bool loadMedia()
{
	//Loading success flag
	bool success = true;

	//Load default surface
	gKeyPressSurfaces[ KEY_PRESS_SURFACE_DEFAULT ] = loadSurface( "press.bmp" );
	if( gKeyPressSurfaces[ KEY_PRESS_SURFACE_DEFAULT ] == NULL )
	{
		printf( "Failed to load default image!\n" );
		success = false;
	}

	//Load up surface
	gKeyPressSurfaces[ KEY_PRESS_SURFACE_UP ] = loadSurface( "up.bmp" );
	if( gKeyPressSurfaces[ KEY_PRESS_SURFACE_UP ] == NULL )
	{
		printf( "Failed to load up image!\n" );
		success = false;
	}

	//Load down surface
	gKeyPressSurfaces[ KEY_PRESS_SURFACE_DOWN ] = loadSurface( "down.bmp" );
	if( gKeyPressSurfaces[ KEY_PRESS_SURFACE_DOWN ] == NULL )
	{
		printf( "Failed to load down image!\n" );
		success = false;
	}

	//Load left surface
	gKeyPressSurfaces[ KEY_PRESS_SURFACE_LEFT ] = loadSurface( "left.bmp" );
	if( gKeyPressSurfaces[ KEY_PRESS_SURFACE_LEFT ] == NULL )
	{
		printf( "Failed to load left image!\n" );
		success = false;
	}

	//Load right surface
	gKeyPressSurfaces[ KEY_PRESS_SURFACE_RIGHT ] = loadSurface( "right.bmp" );
	if( gKeyPressSurfaces[ KEY_PRESS_SURFACE_RIGHT ] == NULL )
	{
		printf( "Failed to load right image!\n" );
		success = false;
	}

	return success;
}

void close()
{

	//Destroy window	
	SDL_DestroyRenderer( gRenderer );
	SDL_DestroyWindow( gWindow );
	gWindow = NULL;
	gRenderer = NULL;

	//Quit SDL subsystems
	IMG_Quit();
	SDL_Quit();
        
	//Deallocate surfaces
	for( int i = 0; i < KEY_PRESS_SURFACE_TOTAL; ++i )
	{
		SDL_FreeSurface( gKeyPressSurfaces[ i ] );
		gKeyPressSurfaces[ i ] = NULL;
	}
}

SDL_Texture* loadTexture( std::string path )
{
    //The final texture
    SDL_Texture* newTexture = NULL;

    //Load image at specified path
    SDL_Surface* loadedSurface = IMG_Load( path.c_str() );
    if( loadedSurface == NULL )
    {
        printf( "Unable to load image %s! SDL_image Error: %s\n", path.c_str(), IMG_GetError() );
    }
    else
    {
        //Create texture from surface pixels
        newTexture = SDL_CreateTextureFromSurface( gRenderer, loadedSurface );
        if( newTexture == NULL )
        {
            printf( "unable to create texture from %s! sdl error: %s\n", path.c_str(), SDL_GetError() );
        }

        //Get rid of old loaded surface
        SDL_FreeSurface( loadedSurface );
    }

    return newTexture;
}

SDL_Surface* loadSurface( std::string path )
{
	//Load image at specified path
	SDL_Surface* loadedSurface = SDL_LoadBMP( path.c_str() );
	if( loadedSurface == NULL )
	{
		printf( "Unable to load image %s! SDL Error: %s\n", path.c_str(), SDL_GetError() );
	}

	return loadedSurface;
}

SDL_Rect createPoint(int xUpLeft, int yUpLeft, int pointDim) 
{
    SDL_Rect point;
    point.x = xUpLeft - pointDim / 2;
    point.y = yUpLeft - pointDim / 2;
    point.h = pointDim;
    point.w = pointDim;

    return(point);

}

void DrawLine(SDL_Renderer* renderer, KeyPoint pt1, KeyPoint pt2)
{
    SDL_RenderDrawLine(renderer, 
            pt1.pt.x, pt1.pt.y, 
            pt2.pt.x, pt2.pt.y);
}

void DrawLineSHIFT(SDL_Renderer* renderer, KeyPoint pt1, int sht1, KeyPoint pt2, int sht2)
{
    SDL_RenderDrawLine(renderer, 
            pt1.pt.x + sht1, pt1.pt.y, 
            pt2.pt.x + sht2, pt2.pt.y);
}



#include <vector>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/ocl.hpp>

using std::cout;
using std::cerr;
using std::vector;
using std::string;

using cv::Mat;
using cv::Point2f;
using cv::KeyPoint;
using cv::Scalar;
using cv::Ptr;

using cv::FastFeatureDetector;
using cv::SimpleBlobDetector;

using cv::DMatch;
using cv::BFMatcher;
using cv::DrawMatchesFlags;
using cv::Feature2D;
using cv::ORB;
using cv::BRISK;
using cv::AKAZE;
using cv::KAZE;

using cv::xfeatures2d::BriefDescriptorExtractor;
using cv::xfeatures2d::SURF;
using cv::SIFT;
using cv::xfeatures2d::DAISY;
using cv::xfeatures2d::FREAK;

const double kDistanceCoef = 4.0;
const int kMaxMatchingSize = 50;

inline void detect_and_compute(string type, Mat& img, vector<KeyPoint>& kpts, Mat& desc) {
    if (type.find("fast") == 0) {
        type = type.substr(4);
        Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(10, true);
        detector->detect(img, kpts);
    }

    if (type.find("blob") == 0) {
        type = type.substr(4);
        Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create();
        detector->detect(img, kpts);
    }
    if (type == "orb") {
        Ptr<ORB> orb = ORB::create();
        orb->detectAndCompute(img, Mat(), kpts, desc);
    }
    if (type == "brisk") {
        Ptr<BRISK> brisk = BRISK::create();
        brisk->detectAndCompute(img, Mat(), kpts, desc);
    }
    if (type == "kaze") {
        Ptr<KAZE> kaze = KAZE::create();
        kaze->detectAndCompute(img, Mat(), kpts, desc);
    }
    if (type == "akaze") {
        Ptr<AKAZE> akaze = AKAZE::create();
        akaze->detectAndCompute(img, Mat(), kpts, desc);
    }
}

inline void match(string type, Mat& desc1, Mat& desc2, vector<DMatch>& matches) {
    matches.clear();
    if (type == "bf") {
        BFMatcher desc_matcher(cv::NORM_L2, true);
        desc_matcher.match(desc1, desc2, matches, Mat());
    }
    if (type == "knn") {
        BFMatcher desc_matcher(cv::NORM_L2, true);
        vector< vector<DMatch> > vmatches;
        desc_matcher.knnMatch(desc1, desc2, vmatches, 1);
        for (int i = 0; i < static_cast<int>(vmatches.size()); ++i) {
            if (!vmatches[i].size()) {
                continue;
            }
            matches.push_back(vmatches[i][0]);
        }
    }
    std::sort(matches.begin(), matches.end());
    while (matches.front().distance * kDistanceCoef < matches.back().distance) {
        matches.pop_back();
    }
    while (matches.size() > kMaxMatchingSize) {
        matches.pop_back();
    }
}


inline void findKeyPointsHomography(vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2,
        vector<DMatch>& matches, vector<char>& match_mask) {
    if (static_cast<int>(match_mask.size()) < 3) {
        return;
    }
    vector<Point2f> pts1;
    vector<Point2f> pts2;
    for (int i = 0; i < static_cast<int>(matches.size()); ++i) {
        pts1.push_back(kpts1[matches[i].queryIdx].pt);
        pts2.push_back(kpts2[matches[i].trainIdx].pt);
    }
    findHomography(pts1, pts2, cv::RANSAC, 4, match_mask);
}


inline void FdetectAndCompute(cv::Mat imBN, string method, vector<KeyPoint>& keyPoints, cv::Mat& descriptors)
{
    if ( method == "orb" )
    {
        Ptr<ORB> orb = ORB::create();
        orb->detectAndCompute(imBN, Mat(), keyPoints, descriptors);
    }
    if ( method == "sift" )
    {
        Ptr<SIFT> sift = cv::SIFT::create();
        sift->detectAndCompute(imBN, cv::Mat(), keyPoints, descriptors);
    }
    if ( method == "fast" )
    {
        Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(10, true);
        detector->detect(imBN, keyPoints);
    }
    if ( method == "surf" ) 
    {
        Ptr<Feature2D> surf = SURF::create(800.0);
        surf->detectAndCompute(imBN, Mat(), keyPoints, descriptors);
    }
    if (method == "brisk") {
        Ptr<BRISK> brisk = BRISK::create();
        brisk->detectAndCompute(imBN, Mat(), keyPoints, descriptors);
    }
    if (method == "kaze") {
        Ptr<KAZE> kaze = KAZE::create();
        kaze->detectAndCompute(imBN, Mat(), keyPoints, descriptors);
    }
    if (method == "akaze") {
        Ptr<AKAZE> akaze = AKAZE::create();
        akaze->detectAndCompute(imBN, Mat(), keyPoints, descriptors);
    }
    if (method == "freak") {
        Ptr<FREAK> freak = FREAK::create();
        freak->compute(imBN, keyPoints, descriptors);
    }
    if (method == "daisy") {
        Ptr<DAISY> daisy = DAISY::create();
        daisy->compute(imBN, keyPoints, descriptors);
    }
    if (method == "brief") {
        Ptr<BriefDescriptorExtractor> brief = BriefDescriptorExtractor::create(64);
        brief->compute(imBN, keyPoints, descriptors);
    }
    /*
    switch( method )
    {
        case "orb":
            Ptr<ORB> orb = ORB::create();
            orb->detectAndCompute(imBN, Mat(), keyPoints, descriptors);
            break;
        case "surf":
            Ptr<Feature2d> sift = SIFT::create();
            sift->detectAndCompute(imBN, cv::Mat(), keyPoints, descriptors);
            break;
        default:
            // do nothing
    }
    */
}

inline void Fmatch(cv::Mat descriptorsCurrent, cv::Mat descriptorsPrevius, std::vector<DMatch>& matches)
{

    //cv::BFMatcher matcher(BRUTEFORCE_HAMMING, true); 
    //cv::BFMatcher matcher(cv::NORM_L2, true); 
    cv::BFMatcher matcher; 
    std::vector<std::vector<DMatch>> knn_matches;
    matcher.knnMatch(descriptorsCurrent, descriptorsPrevius, knn_matches, 2);

    for ( int id = 0; id < knn_matches.size(); id++)
    {
        if(!knn_matches[id].size())
        {
            continue;
        }

        matches.push_back(knn_matches[id][0]);
    }

    std::sort(matches.begin(), matches.end());
    
    while (matches.front().distance * 4.0 < matches.back().distance)
    {
        matches.pop_back();
    }

}

void filterMatch(std::vector<DMatch> matches, std::vector<KeyPoint> currentPointsCollection, std::vector<KeyPoint> previousPointsCollection, std::vector<KeyPoint>& currentPoints, std::vector<KeyPoint>& previousPoints)
{
    for( int id = 0; id < matches.size(); id++)
    {

        int queryIdx = matches[id].queryIdx;
        int trainIdx = matches[id].trainIdx;

        currentPoints.push_back(currentPointsCollection[queryIdx]);
        previousPoints.push_back(previousPointsCollection[trainIdx]);

    }

}

bool essentialFilter(std::vector<DMatch>& matches, std::vector<KeyPoint> currentKeyPoints, std::vector<KeyPoint> previousKeyPoints, cv::Mat kMatrix, cv::Mat *E)
{

    /// chech feature with the essential matrix
    bool isTheKValid = false;
    cv::Mat outlier;
    std::vector<Point2d> curV;
    std::vector<Point2d> preV;
    for(int i = 0; i < currentKeyPoints.size(); i++)
    {
        curV.push_back(currentKeyPoints[i].pt);
    }
    for(int i = 0; i < previousKeyPoints.size(); i++)
    {
        preV.push_back(previousKeyPoints[i].pt);
    }

    std::cout << "curV = " << std::endl << " " << curV << std::endl << std::endl;
    std::cout << "preV = " << std::endl << " " << preV << std::endl << std::endl;
    std::cout << "K = " << std::endl << " " << kMatrix << std::endl << std::endl;

    printf("filtered cur is %d\n", currentKeyPoints.size());
    printf("filtered pre is %d\n", previousKeyPoints.size());
    printf("curV is %d\n", curV.size());
    printf("preV is %d\n", preV.size());

    if ( curV.size() > 4)
    {
        isTheKValid = true;
        cv::Mat Einside =  cv::findEssentialMat(
                curV,
                preV,
                kMatrix,
                RANSAC,
                0.999,
                1.01,
                outlier
                );

        cout << "Einside (inside) = " << endl << " " << Einside << endl << endl;

        *E = Einside;

        cout << "outliers = " << endl << " " << outlier << endl << endl;
        cout << "E (inside) = " << endl << " " << *E << endl << endl;

        //cull outlier
        // printf("the row are %d\n", outlier.rows);
        // printf("the cols are %d\n", outlier.cols);
        for ( int i = outlier.rows - 1; i > 0; --i)
        {
            //if(outlier.at<int>(i,0) == 0)
            if(outlier.data[i] == 0)
            {
                // printf("we are here %d\n", i);
                // printf("the our is %d\n", outlier.data[i]);
                // printf("the our is %d\n", outlier.at<int>(0,i));
                matches.erase(matches.begin() + i);

            }
        }
    }

    return(isTheKValid);
}

void blitFeatures(std::vector<DMatch> matches, std::vector<KeyPoint> pointsCollectionCurrent, std::vector<KeyPoint> pointsCollectionPrevius, SDL_Renderer* gRenderer, vector<int> color)
{

    for ( int id = 0; id < matches.size(); id++)
    {
        int imgIdx = matches[id].imgIdx; int queryIdx = matches[id].queryIdx;
        int trainIdx = matches[id].trainIdx;
        //printf("it is a beautiful img %d, query day %d, train effort %d in the end\n", 
        //        imgIdx, queryIdx, trainIdx);

        // we can ask for k matches...
        //imgIdx = matches[id][1].imgIdx;
        //queryIdx = matches[id][1].queryIdx;
        //trainIdx = matches[id][1].trainIdx;
        //printf("it is a beautiful img %d, query day %d, train effort %d in the end\n", 
        //        imgIdx, queryIdx, trainIdx);
        //

        //draw line first image
        DrawLine(gRenderer, pointsCollectionCurrent[queryIdx], pointsCollectionPrevius[trainIdx]);

        //draw line between the second and current image and the third and previus image 
        DrawLineSHIFT(gRenderer, pointsCollectionCurrent[queryIdx], IMG_WIDTH, pointsCollectionPrevius[trainIdx], IMG_WIDTH * 2 );

        cv::KeyPoint thisPoint = pointsCollectionCurrent[queryIdx];
        // check the size, octave and rotation of the point
        //printf("The point with id %d, size %f, octave %d, and rotation %f\n", queryIdx, thisPoint.size, thisPoint.octave, thisPoint.angle);

        // draw the feature dimension
        rectangleRGBA (
                gRenderer,
                thisPoint.pt.x - (int)thisPoint.size,
                thisPoint.pt.y - (int)thisPoint.size,
                thisPoint.pt.x + (int)thisPoint.size,
                thisPoint.pt.y + (int)thisPoint.size,
                color[0], color[1], color[2], color[3]);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // draw line with thickness with sdl_gfx
        //

        thickLineRGBA( 	
                gRenderer,
                pointsCollectionPrevius[trainIdx].pt.x,
                pointsCollectionPrevius[trainIdx].pt.y,
                pointsCollectionCurrent[queryIdx].pt.x,
                pointsCollectionCurrent[queryIdx].pt.y,
                5,
                color[0], color[1], color[2], color[3]);
        thickLineRGBA( 	
                gRenderer,
                pointsCollectionPrevius[trainIdx].pt.x + IMG_WIDTH * 2,
                pointsCollectionPrevius[trainIdx].pt.y,
                pointsCollectionCurrent[queryIdx].pt.x + IMG_WIDTH,
                pointsCollectionCurrent[queryIdx].pt.y,
                5,
                color[0], color[1], color[2], color[3]);

    }
}

int main( int argc, char** argv )
{

    // poses array
    std::vector<cv::Vec3d> poses;

    //Start up SDL and create a window
    if ( !init() )
    {
        printf( "Failed to load media!\n");
    }
    else
    { 
        /// INTRINSIC  ///////
        float fx = 529.0;
        float fy = 529.0;
        float cx = 320.0;
        float cy = 240.0;

        float dataK[9] = {fx, 0, cx, 0, fy, cy, 0, 0, 1};
        cv::Mat K = cv::Mat(3, 3, CV_32F, dataK); 


        cout << "K = " << endl << " " << K << endl << endl;

        cv::String path("/home/arcblock/datasets/WUST/RGBD-return/rgb-opencvSLAM/*.png"); //select only jpg vector<cv::String> fn;
       // cv::String path("/home/arcblock/datasets/WUST/RGBD/motionblur/*.png"); //select only jpg vector<cv::String> fn;
        vector<cv::String> fn;
        vector<cv::Mat> data;

        printf("hello world\n");

        /////////// MAIN LOOP ///////////////
        bool firstFrame = true;
        cv::glob(path,fn,true); // recurse
        cv::Mat imPrev = cv::imread(fn[0]);
        cv::Mat im; 
        poses.push_back(cv::Vec3d(0.0f, 0.0f, 0.0f));

        for (int k=0; k<135; k++)
        {
            printf("lottomat: %d\n", k);
            if ( !firstFrame )
            {
                imPrev = im;
            }
            im = cv::imread(fn[k]);

            if (im.empty()) continue; //only proceed if sucsessful
            // OPENCV VIEWR // imshow("Display window", im); 

            /*
            /// print im data ///
            printf("number of columns %d\n", im.cols);
            printf("number of rows %d\n", im.rows);
            printf("depth %d\n", im.depth());
            printf("nothing dd %d\n", *im.step.buf);
            printf("here 33\n");
            printf("lottomat2: %d\n", k);
            */

            // BN ////////////////
            cv::Mat imBN;
            cvtColor(im, imBN, cv::COLOR_RGB2GRAY);

            // COPIED FUNCTION
            detect_and_compute("akaze", imBN, M3keyPointsCollection[k], M3descriptorCollection[k]);

            /////////////////////////////////////
            //ORB extraction
            ////////////////////////////////////

            
            FdetectAndCompute(imBN, "orb", keyPointsCollection[0][k], descriptorsCollection[0][k]);
            FdetectAndCompute(imBN, "fast", keyPointsCollection[1][k], descriptorsCollection[1][k]);
            FdetectAndCompute(imBN, "akaze", keyPointsCollection[2][k], descriptorsCollection[2][k]);

            cv::Mat outCorners;
            vector<Point2f> ff;
            vector<KeyPoint> KKff;
            cv::goodFeaturesToTrack(imBN, outCorners, 10, 0.01, 3); 
            //cv::goodFeaturesToTrack(imBN, KKff, 10, 0.01, 3); 
            cv::goodFeaturesToTrack(imBN, ff, 0, 0.01, 3); 

            //cout << "M = " << endl << " " << im << endl << endl;
            //cout << "out = " << endl << " " << outCorners << endl << endl;
            //cout << "outFF = " << endl << " " << ff << endl << endl;

            // convert Point2f to KeyPoint
            vector<KeyPoint> gfttVec;
            for (size_t i = 0; i < ff.size(); i++)
            {
                //gfttVec.push_back(cv::KeyPoint(ff[i], 20));
                MkeyPointsCollection[k].push_back(cv::KeyPoint(ff[i], 20));
            }

            // convert mat in KeyPoint 
            vector<Point2f> outCornersVec; 
            int bb = 0;
            for  (size_t i = 0; i < outCorners.rows; i++) 
            {
                printf("bb = %d\n", bb);
                printf("bb = %d\n", outCorners.rows);
                outCornersVec.push_back(outCorners.at<cv::Point2f>(i));
                //MkeyPointsCollection[k].push_back(outCorners.at<cv::KeyPoint>(i));
                bb++;
            }

            cout << "outFF = " << endl << " " << outCornersVec << endl << endl;
            //ORB extraction for good features

            cout << "descriptors = " << endl << " " << MdescriptorCollection[k] << endl << endl;

            //SDL part
            SDL_Surface* frameSurface = SDL_CreateRGBSurfaceFrom((void*)im.data,
                    640, 480, //im.cols, im.rows,
                    24, // im.depth, is a ref in opencv
                    640 * 3,// *im.step.buf,
                    0x00ff0000, 0x0000ff00, 0x000000ff, 0);

            SDL_Surface* frameSurfacePrev = SDL_CreateRGBSurfaceFrom((void*)imPrev.data,
                    640, 480, //im.cols, im.rows,
                    24, // im.depth, is a ref in opencv
                    640 * 3,// *im.step.buf,
                    0x00ff0000, 0x0000ff00, 0x000000ff, 0);

            if( frameSurface == NULL )
            {
                printf( "Unable to load image %s! SDL Error: %s\n", "rgb from mat", SDL_GetError() );
            }

            printf("here 3\n");

            SDL_Texture* newTexture = NULL;
            SDL_Texture* newTexturePrev = NULL;


            //Create texture from surface pixels
            newTexture = SDL_CreateTextureFromSurface( gRenderer, frameSurface);
            newTexturePrev = SDL_CreateTextureFromSurface( gRenderer, frameSurfacePrev);
            if( newTexture == NULL && newTexturePrev == NULL)
            {
                printf( "unable to create texture from %s! sdl error: %s\n", path.c_str(), SDL_GetError() );
            }

            //Get rid of old loaded surface
            SDL_FreeSurface( frameSurface );

            SDL_Rect points[20];


            //this make a core dump
            //SDL_Surface filledSurface = *gScreenSurface;
            //SDL_SetSurfaceColorMod(filledSurface, 255, 0, 0);

            //Render texture to screen

            //First image
            SDL_Rect DestR;

            DestR.x = 0;
            DestR.y = 0;
            DestR.w = IMG_WIDTH;
            DestR.h = IMG_HEIGHT;

            SDL_RenderCopy( gRenderer, newTexture, NULL, &DestR);

            //Second image
            DestR.x = IMG_WIDTH;
            DestR.y = 0;
            DestR.w = IMG_WIDTH;
            DestR.h = IMG_HEIGHT;

            SDL_RenderCopy( gRenderer, newTexture, NULL, &DestR);


            //Third image
            DestR.x = IMG_WIDTH*2;
            DestR.y = 0;
            DestR.w = IMG_WIDTH;
            DestR.h = IMG_HEIGHT;

            SDL_RenderCopy( gRenderer, newTexturePrev, NULL, &DestR);


            for (int idx = 0; idx < keyPointsCollection[0][k].size(); idx++)
            {
               int d = 3;
               printf("x coor %d, idx: %d, k: %d\n", keyPointsCollection[0][k][idx].pt.x, idx, k);
               SDL_Rect pt = createPoint(keyPointsCollection[0][k][idx].pt.x, keyPointsCollection[0][k][idx].pt.y, 5); 
                           




               //SDL_SetRenderDrawColor( gRenderer, 0x00, 0x00, 0xFF, 0xFF );		
               //SDL_RenderFillRect( gRenderer, &pt);
            }

            //SDL_RenderDrawLine(KRenderer, 20, 20, 400, 400);
            
            //-- Step 2: Matching descriptor vectors with a FLANN based matcher
            // Since SURF is a floating-point descriptor NORM_L2 is used
            std::vector< std::vector<DMatch> > knn_matches;
            std::vector<DMatch> MMmatches;
            if ( !firstFrame )
            {
                // NEW MATCH ///////////////////
                match("knn", M3descriptorCollection[k], M3descriptorCollection[k - 1], MMmatches);

                //Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

                std::vector<DMatch> matches[10];
                cv::Mat E[10];
                //for( int i = 0; i < 10; i++)
                //{
                //    E[i] = cv::Mat();
                //}

                Fmatch(descriptorsCollection[0][k], descriptorsCollection[0][k-1], matches[0]);
                //Fmatch(descriptorsCollection[1][k], descriptorsCollection[1][k-1], matches[1]);
                Fmatch(descriptorsCollection[2][k], descriptorsCollection[2][k-1], matches[2]);
                

                filterMatch(matches[0], keyPointsCollection[0][k], keyPointsCollection[0][k - 1], currentFilteredPointsCollection[0][k], previousFilteredPointsCollection[0][k]);
                filterMatch(matches[2], keyPointsCollection[2][k], keyPointsCollection[2][k - 1], currentFilteredPointsCollection[2][k], previousFilteredPointsCollection[2][k]);


                printf("matches for 0 are %d\n", matches[0].size());
                if(!essentialFilter(matches[0], currentFilteredPointsCollection[0][k], previousFilteredPointsCollection[0][k], K, &E[0]))
                {
                    matches[0].erase(matches[0].begin(), matches[0].end());

                }
                printf("matches for 0 after filtration are %d\n", matches[0].size());
                if(!essentialFilter(matches[2], currentFilteredPointsCollection[2][k], previousFilteredPointsCollection[2][k], K, &E[2]))
                {
                    matches[2].erase(matches[2].begin(), matches[2].end());
                }

                vector<int> color = {255, 0, 255, 120};
                blitFeatures(matches[0], keyPointsCollection[0][k], keyPointsCollection[0][k-1], gRenderer, color);
                color = {255, 255, 0, 180};
                blitFeatures(matches[2], keyPointsCollection[2][k], keyPointsCollection[2][k-1], gRenderer, color);

                // M3 descriptors
                for ( int id = 0; id < MMmatches.size(); id++)
                //for ( int id = 0; id < 4; id++)
                {
                    int imgIdx = MMmatches[id].imgIdx;
                    int queryIdx = MMmatches[id].queryIdx;
                    int trainIdx = MMmatches[id].trainIdx;
                    //printf("it is a beautiful img %d, query day %d, train effort %d in the end\n", 
                    //        imgIdx, queryIdx, trainIdx);

                    SDL_SetRenderDrawColor( gRenderer, 0xFF, 0x00, 0x99, 0xFF );		

                    DrawLine(gRenderer, M3keyPointsCollection[k][queryIdx], M3keyPointsCollection[k-1][trainIdx]);


                }
                
                // the E estimatiojn should work also with only 5 point, but sometimes the algorithm output more the one E packed in a cv::Mat structure
                // Not shure why it do it only sometimes and not always
                if(matches[0].size() >= 6)
                {
                    /// start reconstruction ////
                    /// rt pose ///

                    cv::Mat U[10], D[10], Vt[10];


                    cout << "E = " << endl << " " << E[0] << endl << endl;
                    cv::SVD::compute(E[0], D[0], U[0], Vt[0]);

                    cout << "U = " << endl << " " << U[0] << endl << endl;
                    cout << "D = " << endl << " " << D[0] << endl << endl;
                    cout << "Vt = " << endl << " " << Vt[0] << endl << endl;

                    // following hartley
                    double dataDiag[9] = {1, 0, 0, 0, 1, 0, 0, 0, 0};
                    cv::Mat Diag = cv::Mat(3, 3, CV_64FC1, dataDiag); 

                    double dataW[9] = {0, -1, 0, 1, 0, 0, 0, 0, 1};
                    cv::Mat W = cv::Mat(3, 3, CV_64FC1, dataW); 

                    //cv::Mat S = U[0] * Diag * W * U[0].t();
                    cv::Mat S = Diag * Diag;
                    S = U[0] * Diag * W;
                    S = U[0] * Diag * W * U[0].t();
                    cv::Mat R = U[0] * W * Vt[0];

                    double detUv = cv::determinant(U[0] * Vt[0]);

                    cout << "S = " << endl << " " << S << endl << endl;
                    cout << "R = " << endl << " " << R << endl << endl;

                    cout << "det(U * Vt) " << detUv << endl;

                    // store only the translation of the poses
                    cv::Vec3d t(S.at<double>(2,1), S.at<double>(0,2), S.at<double>(1,0));
                    std::cout << "t = " << std::endl << " " << t << std::endl << std::endl;

                    poses.push_back(poses.back() + t);

                }

                

                /*
                // part from the example
                cv::Mat imP = cv::imread(fn[k - 1]);
                vector<char> match_mask(MMmatches.size(), 1);
                printf(" printiamo the number of mathes %d\n", MMmatches.size());

                if(MMmatches.size() > 3)
                {

                    findKeyPointsHomography(M3keyPointsCollection[k], M3keyPointsCollection[k - 1], 
                            MMmatches, match_mask);

                    Mat res;
                    cv::drawMatches(im, M3keyPointsCollection[k], imP, M3keyPointsCollection[k - 1], 
                            MMmatches, res, Scalar::all(-1),
                            Scalar::all(-1), match_mask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

                    cv::imshow("result", res);
                }
                else
                {
                    Mat res;
                    cv::drawMatches(im, M3keyPointsCollection[k], imP, M3keyPointsCollection[k - 1], 
                            MMmatches, res, Scalar::all(-1),
                            Scalar::all(-1), match_mask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

                    cv::imshow("result", res);
                }
                */

            }
/*
            points[i] = createPoint(keyPointsCollection[i].pt.x, keyPointsCollection[i].pt.y, 4); 
            //SDL_BlitSurface(&filledSurface, point[i], frameSurface, NULL);

            // test rect blit
             SDL_Rect pt = createPoint(300, 200, 100); 
             SDL_BlitSurface(&filledSurface, &pt, frameSurface, NULL);
*/



            /* 
            //Load image at specified path
            SDL_Surface* loadedSurface = SDL_LoadBMP( "scree.bmp" );
            
            SDL_Surface* loae = SDL_CreateRGBSurfaceFrom((void*)loadedSurface->pixels,
                    640, 480, //im.cols, im.rows,
                    32, // im.depth, is a ref in opencv
                    640 * 4,// *im.step.buf,
                    0x00ff0000, 0x0000ff00, 0x000000ff, 0);

            if( loae == NULL )
            {
                printf( "Unable to load image %s! SDL Error: %s\n", "rgb from loaded", SDL_GetError() );
            }

            if( loadedSurface == NULL )
            {
                printf( "Unable to load image %s! SDL Error: %s\n", "scree.bmp", SDL_GetError() );
            }

            */





            //Clear screen
            //SDL_SetRenderDrawColor( gRenderer, 0xFF, 0xFF, 0xFF, 0xFF );
            //SDL_RenderClear( gRenderer );

            //Update screen
            SDL_RenderPresent( gRenderer );

            /*
            //Apply the current image
            SDL_BlitSurface( frameSurface, NULL, gScreenSurface, NULL );
            //SDL_BlitSurface( loadedSurface, NULL, gScreenSurface, NULL );
			
            //Update the surface
            SDL_UpdateWindowSurface( gWindow );
            */

            //printf(typeid(im).name());
            //cout << typeid(im).name() << endl;
            //WAIT KEY FOR THE OPENCV VIEWER ________________ // waitKey();
            // you probably want to do some preprocessing
            data.push_back(im);
            firstFrame = false;
        }
    }
    
    //Free resources and close SDL
    close();

    ///////// start raylib ////////////////////

    // Initialization
    //--------------------------------------------------------------------------------------
    const int screenWidth = 800;
    const int screenHeight = 450;

    InitWindow(screenWidth, screenHeight, "slam 3d viewer");

    // Define the camera to look into our 3d world
    Camera3D camera = { 0 };
    camera.position = (Vector3){ 10.0f, 10.0f, 10.0f }; // Camera position
    camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };      // Camera looking at point
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };          // Camera up vector (rotation towards target)
    camera.fovy = 45.0f;                                // Camera field-of-view Y
    camera.type = CAMERA_PERSPECTIVE;                   // Camera mode type

    Vector3 cubePosition = { 0.0f, 0.0f, 0.0f };

    SetCameraMode(camera, CAMERA_FREE); // Set a free camera mode

    SetTargetFPS(60);               // Set our game to run at 60 frames-per-second
    //--------------------------------------------------------------------------------------

    // Main game loop
    while (!WindowShouldClose())    // Detect window close button or ESC key
    {
        // Update
        //----------------------------------------------------------------------------------
        if (IsKeyPressed(KEY_SPACE))
        {
            if (camera.type == CAMERA_PERSPECTIVE)
            {
                camera.fovy = WIDTH_ORTHOGRAPHIC;
                camera.type = CAMERA_ORTHOGRAPHIC;
            }
            else
            {
                camera.fovy = FOVY_PERSPECTIVE;
                camera.type = CAMERA_PERSPECTIVE;
            }
        }

        UpdateCamera(&camera);          // Update camera

        if (IsKeyDown('Z')) camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };
        //----------------------------------------------------------------------------------

        // Draw
        //----------------------------------------------------------------------------------
        BeginDrawing();

            ClearBackground(RAYWHITE);

            BeginMode3D(camera);
            
            // loop to draw all the pose
            for(int i = 0; i < poses.size(); i++)
            {
                cv::Vec3d v = poses[i];
                DrawSphere((Vector3){v[0], v[1], v[2]}, 0.02f, RED);
                std::cout << "vec = " << v << std::endl;
            }

                DrawGrid(10, 1.0f);        // Draw a grid

            EndMode3D();

            DrawText("I am a SLAM, a real one!", 10, GetScreenHeight() - 30, 20, DARKGRAY);

            if (camera.type == CAMERA_ORTHOGRAPHIC) DrawText("ORTHOGRAPHIC", 10, 40, 20, BLACK);
            else if (camera.type == CAMERA_PERSPECTIVE) DrawText("PERSPECTIVE", 10, 40, 20, BLACK);

            DrawFPS(10, 10);

        EndDrawing();
        //----------------------------------------------------------------------------------
    }

    // De-Initialization
    //--------------------------------------------------------------------------------------
    CloseWindow();        // Close window and OpenGL context
    //--------------------------------------------------------------------------------------

    return 0;
}

/*	
  cv::CommandLineParser parser(argc, argv,
                               "{@input   |lena.jpg|input image}"
                               "{ksize   k|1|ksize (hit 'K' to increase its value at run time)}"
                               "{scale   s|1|scale (hit 'S' to increase its value at run time)}"
                               "{delta   d|0|delta (hit 'D' to increase its value at run time)}"
                               "{help    h|false|show help message}");
  cout << "The sample uses Sobel or Scharr OpenCV functions for edge detection\n\n";
  parser.printMessage();
  cout << "\nPress 'ESC' to exit program.\nPress 'R' to reset values ( ksize will be -1 equal to Scharr function )";
  // First we declare the variables we are going to use
  Mat image,src, src_gray;
  Mat grad;
  const String window_name = "Sobel Demo - Simple Edge Detector";
  int ksize = parser.get<int>("ksize");
  int scale = parser.get<int>("scale");
  int delta = parser.get<int>("delta");
  int ddepth = CV_16S;
  String imageName = parser.get<String>("@input");
  // As usual we load our source image (src)
  image = imread( samples::findFile( imageName ), IMREAD_COLOR ); // Load an image
  // Check if image is loaded fine
  if( image.empty() )
  {
    printf("Error opening image: %s\n", imageName.c_str());
    return EXIT_FAILURE;
  }
  for (;;)
  {
    // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
    GaussianBlur(image, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
    // Convert the image to grayscale
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Sobel(src_gray, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
    Sobel(src_gray, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
    // converting back to CV_8U
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
    imshow(window_name, grad);
    char key = (char)waitKey(0);
    if(key == 27)
    {
      return EXIT_SUCCESS;
    }
    if (key == 'k' || key == 'K')
    {
      ksize = ksize < 30 ? ksize+2 : -1;
    }
    if (key == 's' || key == 'S')
    {
      scale++;
    }
    if (key == 'd' || key == 'D')
    {
      delta++;
    }
    if (key == 'r' || key == 'R')
    {
      scale =  1;
      ksize = -1;
      delta =  0;
    }
  }
  return EXIT_SUCCESS;
  */
