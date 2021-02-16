// STL
#include <iostream>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <chrono>

// openCV
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/ocl.hpp>

#include <opencv2/objdetect.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d.hpp>

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

#define PRINT_MAT 0

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

//Frees media and shuts down SDL
void close();

//The window we'll be rendering to
SDL_Window* gWindow = NULL;
	
//The images that correspond to a keypress
SDL_Surface* gKeyPressSurfaces[ KEY_PRESS_SURFACE_TOTAL ];

//Current displayed image
SDL_Surface* gCurrentSurface = NULL;

//Current rendere
SDL_Renderer* gRenderer = NULL;

//features data
std::vector<cv::KeyPoint> keyPointsCollection[10][200];
cv::Mat descriptorsCollection[10][200];
std::vector<cv::KeyPoint> currentFilteredPointsCollection[10][200];
std::vector<cv::KeyPoint> previousFilteredPointsCollection[10][200];

std::vector<cv::KeyPoint> MkeyPointsCollection[200];
cv::Mat MdescriptorCollection[200];
std::vector<cv::KeyPoint> M2keyPointsCollection[200];
cv::Mat M2descriptorCollection[200];
std::vector<cv::KeyPoint> M3keyPointsCollection[200];
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

SDL_Rect createPoint(int xUpLeft, int yUpLeft, int pointDim) 
{
    SDL_Rect point;
    point.x = xUpLeft - pointDim / 2;
    point.y = yUpLeft - pointDim / 2;
    point.h = pointDim;
    point.w = pointDim;

    return(point);

}

void DrawLine(SDL_Renderer* renderer, cv::KeyPoint pt1, cv::KeyPoint pt2)
{
    SDL_RenderDrawLine(renderer, 
            pt1.pt.x, pt1.pt.y, 
            pt2.pt.x, pt2.pt.y);
}

void DrawLineSHIFT(SDL_Renderer* renderer, cv::KeyPoint pt1, int sht1, cv::KeyPoint pt2, int sht2)
{
    SDL_RenderDrawLine(renderer, 
            pt1.pt.x + sht1, pt1.pt.y, 
            pt2.pt.x + sht2, pt2.pt.y);
}

void blitFeatures(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> pointsCollectionCurrent, std::vector<cv::KeyPoint> pointsCollectionPrevius, SDL_Renderer* gRenderer,std::vector<int> color)
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



inline void FdetectAndCompute(cv::Mat imBN, std::string method, std::vector<cv::KeyPoint>& keyPoints, cv::Mat& descriptors)
{
    if ( method == "orb" )
    {
        cv::Ptr<cv::ORB> orb = cv::ORB::create();
        orb->detectAndCompute(imBN, cv::Mat(), keyPoints, descriptors);
    }
    if ( method == "sift" )
    {
        cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
        sift->detectAndCompute(imBN, cv::Mat(), keyPoints, descriptors);
    }
    if ( method == "fast" )
    {
        cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(10, true);
        detector->detect(imBN, keyPoints);
    }
    if ( method == "surf" ) 
    {
        cv::Ptr<cv::Feature2D> surf = cv::xfeatures2d::SURF::create(800.0);
        surf->detectAndCompute(imBN, cv::Mat(), keyPoints, descriptors);
    }
    if (method == "brisk") {
        cv::Ptr<cv::BRISK> brisk = cv::BRISK::create();
        brisk->detectAndCompute(imBN, cv::Mat(), keyPoints, descriptors);
    }
    if (method == "kaze") {
        cv::Ptr<cv::KAZE> kaze = cv::KAZE::create();
        kaze->detectAndCompute(imBN, cv::Mat(), keyPoints, descriptors);
    }
    if (method == "akaze") {
        cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
        akaze->detectAndCompute(imBN, cv::Mat(), keyPoints, descriptors);
    }
    if (method == "freak") {
        cv::Ptr<cv::xfeatures2d::FREAK> freak = cv::xfeatures2d::FREAK::create();
        freak->compute(imBN, keyPoints, descriptors);
    }
    if (method == "daisy") {
        cv::Ptr<cv::xfeatures2d::DAISY> daisy = cv::xfeatures2d::DAISY::create();
        daisy->compute(imBN, keyPoints, descriptors);
    }
    if (method == "brief") {
        cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief = cv::xfeatures2d::BriefDescriptorExtractor::create(64);
        brief->compute(imBN, keyPoints, descriptors);
    }
}

inline void Fmatch(cv::Mat descriptorsCurrent, cv::Mat descriptorsPrevius, std::vector<cv::DMatch>& matches)
{

    //cv::BFMatcher matcher(BRUTEFORCE_HAMMING, true); 
    //cv::BFMatcher matcher(cv::NORM_L2, true); 
    cv::BFMatcher matcher; 
    std::vector<std::vector<cv::DMatch>> knn_matches;
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

void filterMatch(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> currentPointsCollection, std::vector<cv::KeyPoint> previousPointsCollection, std::vector<cv::KeyPoint>& currentPoints, std::vector<cv::KeyPoint>& previousPoints)
{
    for( int id = 0; id < matches.size(); id++)
    {

        int queryIdx = matches[id].queryIdx;
        int trainIdx = matches[id].trainIdx;

        currentPoints.push_back(currentPointsCollection[queryIdx]);
        previousPoints.push_back(previousPointsCollection[trainIdx]);

    }

}

bool essentialFilter(std::vector<cv::DMatch>& matches, std::vector<cv::KeyPoint> currentKeyPoints, std::vector<cv::KeyPoint> previousKeyPoints, cv::Mat kMatrix, cv::Mat *E)
{

    /// chech feature with the essential matrix
    bool isTheKValid = false;
    cv::Mat outlier;
    std::vector<cv::Point2d> curV;
    std::vector<cv::Point2d> preV;
    for(int i = 0; i < currentKeyPoints.size(); i++)
    {
        curV.push_back(currentKeyPoints[i].pt);
    }
    for(int i = 0; i < previousKeyPoints.size(); i++)
    {
        preV.push_back(previousKeyPoints[i].pt);
    }


#if PRINT_MAT
    std::cout << "curV = " << std::endl << " " << curV << std::endl << std::endl;
    std::cout << "preV = " << std::endl << " " << preV << std::endl << std::endl;
    std::cout << "K = " << std::endl << " " << kMatrix << std::endl << std::endl;

    printf("filtered cur is %d\n", currentKeyPoints.size());
    printf("filtered pre is %d\n", previousKeyPoints.size());
    printf("curV is %d\n", curV.size());
    printf("preV is %d\n", preV.size());
#endif

    if ( curV.size() > 4)
    {
        isTheKValid = true;
        cv::Mat Einside =  cv::findEssentialMat(
                curV,
                preV,
               kMatrix,
                cv::RANSAC,
                0.999,
                1.01,
                outlier
                );

        *E = Einside;
#if PRINT_MAT
        std::cout << "Einside (inside) = " << std::endl << " " << Einside << std::endl << std::endl;

        std::cout << "outliers = " << std::endl << " " << outlier << std::endl << std::endl;
        std::cout << "E (inside) = " << std::endl << " " << *E << std::endl << std::endl;
#endif


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

// add n col to a cv::mat
void resizeCol(cv::Mat& m, size_t sz, const cv::Scalar& s)
{
    cv::Mat tm(m.rows, m.cols + sz, m.type());
    tm.setTo(s);
    m.copyTo(tm(cv::Rect(cv::Point(0, 0), m.size())));
    m = tm;
}


////// temp code to find the home directory //////
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>


#include <thread>
#include <mutex>

std::mutex mtx;
std::mutex mtxSDL;

int StartViewer(std::vector<cv::Vec3d> *poses, std::vector<cv::Vec3d> *posesH, std::vector<cv::Mat> *posesHom)
{
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

        cv::Vec3d sum(0,0,0);
        UpdateCamera(&camera);          // Update camera

        if (IsKeyDown('Z')) camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };
        //----------------------------------------------------------------------------------

        // Draw
        //----------------------------------------------------------------------------------
        BeginDrawing();

            ClearBackground(RAYWHITE);

            BeginMode3D(camera);
            
                mtx.lock();
                // loop to draw all the pose
                for(int i = 0; i < poses->size(); i++)
                {
                    cv::Vec3d v = (*poses)[i];
                    DrawSphere((Vector3){v[0], v[1], v[2]}, 0.08f, RED);
                    //std::cout << "vec = " << v << std::endl;
                    cv::Vec3d vH = (*posesH)[i];
                    DrawSphere((Vector3){vH[0], vH[1], vH[2]}, 0.08f, BLUE);

                    //poses drawing
                    cv::Mat M = (*posesHom)[i];
                    cv::Vec3d position(M.at<double>(0,3), M.at<double>(1,3), M.at<double>(2,3));
                    std::cout << "postion = " << std::endl << " " << position << std::endl << std::endl;
                    sum = sum + position;
                    DrawSphere((Vector3){position[0], position[1], position[2]}, 0.04f, ORANGE);
                    DrawSphere((Vector3){sum[0], sum[1], sum[2]}, 0.06f, PINK);
                }
                mtx.unlock();

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

void SDLloop(SDL_Renderer* gRenderer)
{

    std::chrono::milliseconds sleepTime(100); 

    while(1)
    {
        mtxSDL.lock();
        SDL_SetRenderDrawColor( gRenderer, 0x00, 0xFF, 0xFF, 0xFF);
        SDL_RenderPresent(gRenderer);
        printf("i am in the thread?\n");
        mtxSDL.unlock();
        std::this_thread::sleep_for(sleepTime);
    }

}

/////////////////////////////////////




int main( int argc, char** argv )
{

    // poses array
    std::vector<cv::Vec3d> poses;
    std::vector<cv::Vec3d> posesH;
    cv::Vec3d vec(0.0f, 0.0f, 0.0f);
    poses.push_back(vec);
    posesH.push_back(vec);

    std::vector<cv::Mat> posesHom;
    cv::Mat pose0 = cv::Mat::eye(4, 4, CV_64FC1);
    posesHom.push_back(pose0);

    // init sdl
    if(!init())
    {
        printf( "Failed to load SDL!\n");
    }

    // start interfaces, ray and sdl
    std::thread viewer3d(StartViewer,&poses, &posesH, &posesHom);
    std::thread viewer2d(SDLloop, gRenderer);

    /// INTRINSIC  ///////
    float fx = 529.0;
    float fy = 529.0;
    float cx = 320.0;
    float cy = 240.0;

    float dataK[9] = {fx, 0, cx, 0, fy, cy, 0, 0, 1};
    cv::Mat K = cv::Mat(3, 3, CV_32F, dataK); 
#if PRINT_MAT
    std::cout << "K = " << std::endl << " " << K << std::endl << std::endl;
#endif

    //// temp code home folder
    const char *homedir;
    if((homedir = getenv("HOME")) == NULL) 
    {
        homedir = getpwuid(getuid())->pw_dir;
    }
    cv::String home(homedir);
    cv::String folderPath("/datasets/WUST/RGBD-return/rgb-opencvSLAM/*.png"); //select only jpg vector<cv::String> fn;
    cv::String path = home + folderPath;
    std::cout << path << std::endl;
    // cv::String path("/home/arcblock/datasets/WUST/RGBD/motionblur/*.png"); //select only jpg vector<cv::String> fn;
    std::vector<cv::String> fn;
    std::vector<cv::Mat> data;

    /////////// MAIN LOOP ///////////////
    bool firstFrame = true;
    cv::glob(path,fn,true); // recurse
    cv::Mat imPrev = cv::imread(fn[0]);
    cv::Mat im; 

    for (int k=0; k<135; k++)
    {
        if ( !firstFrame )
        {
            imPrev = im;
        }
        im = cv::imread(fn[k]);

        if (im.empty()) continue; //only proceed if sucsessful
        // OPENCV VIEWR // 
        imshow("Display window", im); 

        /*
        /// print im data ///
        printf("number of columns %d\n", im.cols);
        printf("number of rows %d\n", im.rows);
        printf("depth %d\n", im.depth());
        printf("nothing dd %d\n", *im.step.buf);
        */

        // BN ////////////////
        cv::Mat imBN;
        cvtColor(im, imBN, cv::COLOR_RGB2GRAY);

        FdetectAndCompute(imBN, "akaze", M3keyPointsCollection[k], M3descriptorCollection[k]);

        /////////////////////////////////////
        //ORB extraction
        ////////////////////////////////////


        FdetectAndCompute(imBN, "orb", keyPointsCollection[0][k], descriptorsCollection[0][k]);
        FdetectAndCompute(imBN, "fast", keyPointsCollection[1][k], descriptorsCollection[1][k]);
        FdetectAndCompute(imBN, "akaze", keyPointsCollection[2][k], descriptorsCollection[2][k]);

        cv::Mat outCorners;
        std::vector<cv::Point2f> ff;
        std::vector<cv::KeyPoint> KKff;
        cv::goodFeaturesToTrack(imBN, outCorners, 10, 0.01, 3); 
        //cv::goodFeaturesToTrack(imBN, KKff, 10, 0.01, 3); 
        cv::goodFeaturesToTrack(imBN, ff, 0, 0.01, 3); 

        //cout << "M = " << std::endl << " " << im << std::endl << std::endl;
        //cout << "out = " << std::endl << " " << outCorners << std::endl << std::endl;
        //cout << "outFF = " << std::endl << " " << ff << std::endl << std::endl;

        // convert Point2f to KeyPoint
        std::vector<cv::KeyPoint> gfttVec;
        for (size_t i = 0; i < ff.size(); i++)
        {
            //gfttVec.push_back(cv::KeyPoint(ff[i], 20));
            MkeyPointsCollection[k].push_back(cv::KeyPoint(ff[i], 20));
        }

        // convert mat in KeyPoint 
        std::vector<cv::Point2f> outCornersVec; 
        int bb = 0;
        for  (size_t i = 0; i < outCorners.rows; i++) 
        {
            outCornersVec.push_back(outCorners.at<cv::Point2f>(i));
            //MkeyPointsCollection[k].push_back(outCorners.at<cv::KeyPoint>(i));
            bb++;
        }




#if PRINT_MAT
        std::cout << "outFF = " << std::endl << " " << outCornersVec << std::endl << std::endl;
        //ORB extraction for good features

        std::cout << "descriptors = " << std::endl << " " << MdescriptorCollection[k] << std::endl << std::endl;
#endif

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

        SDL_Texture* newTexture = NULL;
        SDL_Texture* newTexturePrev = NULL;

        mtxSDL.lock();
        //Create texture from surface pixels
        newTexture = SDL_CreateTextureFromSurface( gRenderer, frameSurface);
        newTexturePrev = SDL_CreateTextureFromSurface( gRenderer, frameSurfacePrev);
        if( newTexture == NULL && newTexturePrev == NULL)
        {
            printf( "unable to create texture from %s! sdl error: %s\n", path.c_str(), SDL_GetError() );
        }
        mtxSDL.unlock();

        //Get rid of old loaded surface
        SDL_FreeSurface( frameSurface );

        SDL_Rect points[20];

        //Render texture to screen

        //First image
        mtxSDL.lock(); 
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
        mtxSDL.unlock(); 

        int gh = 6;

        int *nn = &gh;

        for (int idx = 0; idx < keyPointsCollection[0][k].size(); idx++)
        {
            int d = 3;
            //printf("x coor %d, idx: %d, k: %d\n", keyPointsCollection[0][k][idx].pt.x, idx, k);
            SDL_Rect pt = createPoint(keyPointsCollection[0][k][idx].pt.x, keyPointsCollection[0][k][idx].pt.y, 5); 
        }

        //-- Step 2: Matching descriptor vectors with a FLANN based matcher
        // Since SURF is a floating-point descriptor NORM_L2 is used
        std::vector< std::vector<cv::DMatch> > knn_matches;
        std::vector<cv::DMatch> MMmatches;
        if ( !firstFrame )
        {
            // NEW MATCH ///////////////////
            Fmatch(M3descriptorCollection[k], M3descriptorCollection[k - 1], MMmatches);

            std::vector<cv::DMatch> matches[10];
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


            //printf("matches for 0 are %d\n", matches[0].size());
            if(!essentialFilter(matches[0], currentFilteredPointsCollection[0][k], previousFilteredPointsCollection[0][k], K, &E[0]))
            {
                matches[0].erase(matches[0].begin(), matches[0].end());

            }
            //printf("matches for 0 after filtration are %d\n", matches[0].size());
            if(!essentialFilter(matches[2], currentFilteredPointsCollection[2][k], previousFilteredPointsCollection[2][k], K, &E[2]))
            {
                matches[2].erase(matches[2].begin(), matches[2].end());
            }


            mtxSDL.lock(); 
            std::vector<int> color = {255, 0, 255, 120};
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
            mtxSDL.unlock(); 

            // the E estimatiojn should work also with only 5 point, but sometimes the algorithm output more the one E packed in a cv::Mat structure
            // Not shure why it do it only sometimes and not always
            if(matches[0].size() >= 6)
            {
                /// start reconstruction ////
                /// rt pose ///

                cv::Mat U[10], D[10], Vt[10];
                cv::SVD::compute(E[0], D[0], U[0], Vt[0]);
#if PRINT_MAT
                std::cout << "E = " << std::endl << " " << E[0] << std::endl << std::endl;
                std::cout << "U = " << std::endl << " " << U[0] << std::endl << std::endl;
                std::cout << "D = " << std::endl << " " << D[0] << std::endl << std::endl;
                std::cout << "Vt = " << std::endl << " " << Vt[0] << std::endl << std::endl;
#endif

                // following hartley
                double dataDiag[9] = {1, 0, 0, 0, 1, 0, 0, 0, 0};
                cv::Mat Diag = cv::Mat(3, 3, CV_64FC1, dataDiag); 

                double dataW[9] = {0, -1, 0, 1, 0, 0, 0, 0, 1};
                cv::Mat W = cv::Mat(3, 3, CV_64FC1, dataW); 

                double dataZ[9] = {0, 1, 0, -1, 0, 0, 0, 0, 0};
                cv::Mat Z = cv::Mat(3, 3, CV_64FC1, dataZ); 

                //cv::Mat S = U[0] * Z * U[0].t();
                cv::Mat S = U[0] * Diag * W * U[0].t();
                cv::Mat R = U[0] * W * Vt[0];

                double detUv = cv::determinant(U[0] * Vt[0]);
                cv::Vec3d t(S.at<double>(2,1), S.at<double>(0,2), S.at<double>(1,0));
                cv::Mat t_mat(t, CV_64FC1);
                cv::Vec3d t_hotz(U[0].at<double>(0,2), U[0].at<double>(1,2), U[0].at<double>(2,2));

                //create a homogeneus pose
                cv::Mat pose(4,4, CV_64FC1);
                pose.setTo(0);
                R.copyTo(pose(cv::Rect(cv::Point(0,0), R.size())));
                t_mat.copyTo(pose.col(3)(cv::Rect(cv::Point(0,0), t_mat.size())));
                pose.at<double>(3,3) = 1.0f;

#if PRINT_MAT
                std::cout << "S = " << std::endl << " " << S << std::endl << std::endl;
                std::cout << "R = " << std::endl << " " << R << std::endl << std::endl;
                std::cout << "det(U * Vt) " << detUv << std::endl;

                // store only the translation of the poses
                std::cout << "t = " << std::endl << " " << t << std::endl << std::endl;
                std::cout << "t_hotz = " << std::endl << " " << t_hotz << std::endl << std::endl;

                std::cout << "pose = " << std::endl << " " << pose << std::endl << std::endl;

#endif
                std::cout << "pose = " << std::endl << " " << pose << std::endl << std::endl;

                mtx.lock();
                poses.push_back(poses.back() + t);
                posesH.push_back(posesH.back() + t_hotz);

                cv::Mat newPose = posesHom.back() * pose;
                posesHom.push_back(newPose);
                mtx.unlock();

                std::cout << "newPose = " << std::endl << " " << newPose << std::endl << std::endl;
                std::cout << "posesHom = " << std::endl << " " << posesHom.back() << std::endl << std::endl;
            }

        }



        //Clear screen
        // mtxSDL.lock(); 
        //SDL_SetRenderDrawColor( gRenderer, 0xFF, 0xFF, 0xFF, 0xFF );
        //SDL_RenderClear( gRenderer );
        // mtxSDL.unlock(); 

        //Update screen
        //SDL_RenderPresent(gRenderer);

        //printf(typeid(im).name());
        //cout << typeid(im).name() << std::endl;
        //WAIT KEY FOR THE OPENCV VIEWER ________________ // 
        cv::waitKey();
        // you probably want to do some preprocessing
        data.push_back(im);
        firstFrame = false;
    }
    
    
    //Free resources and close SDL
    close();

}
