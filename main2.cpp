/********************************************************************************************************************
<For compille and install information>
1. sudo apt-get install curl // sudo apt-get install libcurl4-openssl-dev
2. g++ test.cpp -o d `pkg-config --cflags --libs opencv`
./bin/bow ImageData/Flickr_sub/ data/NUS-WIDE-urls.txt -2 ImageData/testing/volcano_1.jpg volcano_1.yml

                                                                                   <Usage>
1. DOWNLOAD IMAGES
		./bin/bow -1 [URLs.txt] [partial load.txt]
		ex: ./bin/bow -1 data/NUS-WIDE-urls.txt data/NUS-WIDE-OBJECT/image_list/TrainObject_image_name.txt
2. BOW TRAINING
		./bin/bow -2 [FILEs.txt/FILEs Dir] [dictionary.yml]
		ex: ./bin/bow -2 data/NUS-WIDE-OBJECT/image_list/TrainObject_image_name.txt Object_dictionary.yml
		ex: ./bin/bow -2 ImageData/Flickr_sub/ dictionary.yml
3. BOW TESTING
		./bin/bow -3 [image.jpg] [bow.yml] // [FILEs.txt] [bow.yml]
		ex: ./bin/bow -3 ImageData/testing/volcano_1.jpg volcano_1.yml
		ex: ./bin/bow -3 data/NUS-WIDE-OBJECT/image_list/TrainObject_image_name.txt train_bow.txt

4. BOW BRUTE FORCE KNN
		
		./bin/bow -4 Train_bow.txt Test_bow.txt
5. BOW HASH W/ GRAPH KNN
		./bin/bow ImageData/Flickr_sub/ data/NUS-WIDE-urls.txt -2 ImageData/testing/volcano_1.jpg volcano_1.yml

********************************************************************************************************************/
 
// ./bin/bow ImageData/Flickr_sub/ data/NUS-WIDE-urls.txt -2 ImageData/testing/volcano_1.jpg volcano_1.yml



#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>
#include <curl/curl.h>
#include <stdio.h>
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <sys/types.h> 
#include <sys/stat.h>
#include <dirent.h>
#include <math.h>
#include <eigen/Eigen/Dense>

using namespace cv;
using namespace std;
typedef std::map< int, std::vector<string> > nus_file;
using Eigen::MatrixXd;

size_t callbackfunction(void *ptr, size_t size, size_t nmemb, void* userdata)
{
    FILE* stream = (FILE*)userdata;
    if (!stream)
    {
        printf("!!! No stream\n");
        return 0;
    }

    size_t written = fwrite((FILE*)ptr, size, nmemb, stream);
    return written;
}

bool download_jpeg(char* url, char* file_name)
{
//"data/nus_small/out.jpg"
    FILE* fp = fopen(file_name, "wb");
    if (!fp)
    {
        printf("!!! Failed to create file on the disk\n");
        return false;
    }

    CURL* curlCtx = curl_easy_init();
    curl_easy_setopt(curlCtx, CURLOPT_URL, url);
    curl_easy_setopt(curlCtx, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curlCtx, CURLOPT_WRITEFUNCTION, callbackfunction);
    curl_easy_setopt(curlCtx, CURLOPT_FOLLOWLOCATION, 1);

    CURLcode rc = curl_easy_perform(curlCtx);
    if (rc)
    {
    	cout<<"Failed to download: "<<url<<endl;
    	return false;
    }

    long res_code = 0;
    curl_easy_getinfo(curlCtx, CURLINFO_RESPONSE_CODE, &res_code);
    if (!((res_code == 200 || res_code == 201) && rc != CURLE_ABORTED_BY_CALLBACK))
    {
    	cout<<"!!! Response code: "<<res_code<<endl;
    	return false;
    }

    curl_easy_cleanup(curlCtx);

    fclose(fp);

    return true;
}

bool myStr2Int(const string& str, int& num)
{
   num = 0;
   size_t i = 0;
   int sign = 1;
   if (str[0] == '-') { sign = -1; i = 1; }
   bool valid = false;
   for (; i < str.size(); ++i) {
      if (isdigit(str[i])) {
         num *= 10;
         num += int(str[i] - '0');
         valid = true;
      }
      else return false;
   }
   num *= sign;
   return valid;
}

void readdata (fstream &file, nus_file &url_file_all)
{

	// nus_file[photo_id] = url_all
	// url_all = [photo_id_s, photo_file, url_s, url_m, url_l, url_o]
	
	string buffer;
	getline(file,buffer);
	//file>>buffer;
	
	for(int i=1; i<269650; i++){
		// photo_file    Photo_id    url_Large   url_Middle   url_Small  url_Original
		
		getline(file,buffer);
		char buffer_c[700];
		char *pch;
		int photo_id;
		
		
		vector<string> url_all;
		
		strcpy(buffer_c, buffer.c_str());
		pch = strtok(buffer_c, " ");
		string tmp(pch);
		url_all.push_back(tmp);
		pch = strtok(NULL," ");
		string id_s(pch);
		myStr2Int(id_s, photo_id);
		
		while(pch!=NULL){
			//cout<<pch<<endl;
			string tmp(pch);
			url_all.push_back(tmp);
			pch = strtok(NULL," ");
		}
		
		if (url_all.size()!=6) cout<<"err:"<<i<<endl;
		
		buffer.clear();		
		url_file_all[photo_id] = url_all;

		
	}
	return;
}


int main(int argc, char *argv[])
{
 	/*******************************************************************
 								DOWNLOAD IMAGES 
 	********************************************************************/
 	// Photo_file    Photo_id    url_Large   url_Middle   url_Small  url_Origina
	if (argv[1][1]=='1') {
		nus_file url_all;
		fstream file(argv[2]);
		readdata (file, url_all);
		cout<<"Total number of URL is "<<url_all.size()<<endl;
		string file_name, url, dir_name;
		char url_c[100], file_name_c[100], dir_name_c[100];
		int count = 0;
		
		fstream file2(argv[3]);
		vector<int> partial_load;
		string buffer, tmp;
		int id;
		while(getline(file2,buffer)){
			buffer = buffer.substr(1);
			tmp = buffer.substr( buffer.find("\\")+1);
			tmp = tmp.substr( tmp.find("_")+1);
			tmp = tmp.substr(0, tmp.size()-4);
			myStr2Int(tmp, id);
			partial_load.push_back(id);
		}
		cout<<"Number of images download : "<<partial_load.size()<<endl; 
		
		//for (nus_file::iterator it = url_all.begin(); it!= url_all.end(); ++it){
		for (int j=0; j<partial_load.size(); j++) {
			//file_name = it->second[0].substr(3); // C:\ImageData\Flickr\actor\0010_6804082.jpg 
			vector<string> url_s;
			url_s = url_all[partial_load[j]];
			file_name = url_s[0].substr(3);
			
			int size_dir = 0;
			int pos;
			string tmp;
			for (int i=0; i<file_name.size(); i++){
				if (file_name[i]==92) {size_dir++; file_name[i] = 47;}
				pos = i;
				if (size_dir==3) break;
			}
			dir_name = file_name.substr(0,pos);
			dir_name = "mkdir -p "+dir_name;

			//url = it->second[4];
			url =  url_s[4];
			strcpy(url_c, url.c_str());
			strcpy(file_name_c, file_name.c_str());
			strcpy(dir_name_c, dir_name.c_str());
			system( dir_name_c);
	
			if (!download_jpeg(url_c, file_name_c)){
				//cout<<"!! Failed to download file: "<<it->second[1]<<endl; 
				cout<<"!! Failed to download file: "<<url_s[0]<<endl; 
				count++;
				remove(file_name_c);
				continue;
			}
			
		}
		
		cout<<"Number of fail files : "<<count<<endl;
	}
		

 	/*******************************************************************
 								BOW TRAINING
 	********************************************************************/
 	
	else if (argv[1][1]=='2') {
		// read Category: cateName
		DIR *dir;
		vector<string> cateName, tmp;
		vector< vector<string> > imgName;
		vector<string> tmp_all_img;
		
		
		// training image for all image in certain file
		struct dirent *ent;
		if ((dir = opendir (argv[2])) != NULL) {
			// read image category in dir: cateName
		  while ((ent = readdir (dir)) != NULL) {
			 cateName.push_back(ent->d_name);
		  }
		  closedir (dir);
		  	
		  	// read Imagefiles: imageName; store image file name(whole path) in tmp_all_img
			cout<<"Total number of category is "<<cateName.size()<<endl;
			for (int i=2; i<cateName.size(); i++){
			 	DIR *dir2;
				struct dirent *ent2;
				char dirName_c[100];
				string dirName = argv[1]+cateName[i]+"/";
				strcpy(dirName_c, dirName.c_str());
				tmp.push_back(cateName[i]);
				if ( (dir2 = opendir (dirName_c)) != NULL ){
				  while ((ent2 = readdir (dir2)) != NULL) {
					tmp.push_back(ent2->d_name);
					tmp_all_img.push_back(dirName+ent2->d_name);
				  }
				  closedir (dir2);
				} 
				else {
					cout<<"Director error in traning: "<<dirName<<endl;
					continue;
				}
			 	imgName.push_back(tmp);
			 	tmp.clear();
			 }
		} 
		
		// train image in .txt files
		else {
			cout<<"Partial Training..."<<endl;
			fstream file2(argv[2]);
			string buffer;
			while(getline(file2,buffer)){
				buffer = "ImageData/Flickr"+buffer;
				for (int i=0; i<buffer.size(); i++){
					if (buffer[i]==92) buffer[i]=47;
				}
				tmp_all_img.push_back(buffer);
			}
		}		 
		 
		 //Step 1 - Obtain the set of bags of features.
		cout<<"Total number of img is "<<tmp_all_img.size()<<endl;
		int errimg = 0;
		char * filename = new char[100];		
		Mat input;	
		vector<KeyPoint> keypoints;
		Mat descriptor;
		Mat featuresUnclustered; //To store all the descriptors that are extracted from all the images.
		SiftDescriptorExtractor detector;	

		for (int i=0; i<tmp_all_img.size(); i+=1){
			string imgName = tmp_all_img[i];
			strcpy(filename, imgName.c_str());
			input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE); //Load as grayscale	
			if (!input.data){
				//cout<<"error img"<<endl;
				errimg++;
				continue;
			}
			else{
				detector.detect(input, keypoints);
				detector.compute(input, keypoints,descriptor);		
				featuresUnclustered.push_back(descriptor);
			}
			
		}
		cout<<"Total number of error image is "<<errimg<<endl;
		
		//Construct BOWKMeansTrainer
		int dictionarySize=500;
		TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
		int retries=1;
		int flags=KMEANS_PP_CENTERS;
		BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);
		Mat dictionary=bowTrainer.cluster(featuresUnclustered);	
		FileStorage fs(argv[3], FileStorage::WRITE);
		fs << "vocabulary" << dictionary;
		fs.release();
	}
	
	
 	/*******************************************************************
 								BOW TESTING 
 	********************************************************************/
 	
	else if (argv[1][1]=='3') {
		//Step 2 - Obtain the BoF descriptor for given image/video frame. 
		 //prepare BOW descriptor extractor from the dictionary    
		Mat dictionary; 
		FileStorage fs("Object_dictionary.yml", FileStorage::READ);
		fs["vocabulary"] >> dictionary;
		fs.release();	
		 
		Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher); //create a nearest neighbor matcher
		Ptr<FeatureDetector> detector(new SiftFeatureDetector()); //create Sift feature point extracter
		Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);	//create Sift descriptor extractor
		
		BOWImgDescriptorExtractor bowDE(extractor,matcher); //create BoW descriptor extractor
		bowDE.setVocabulary(dictionary); //Set the dictionary with the vocabulary we created in the first step


		char * filename = new char[100];
		char * imageTag = new char[100]; //To store the image tag name - only for save the descriptor in a file

		//open the file to write the resultant descriptor
		//FileStorage fs1(argv[3] , FileStorage::WRITE);	

		fstream file2(argv[2]);
		string buffer,buffer2, tmp;
		char buffer_c[100];
		int count,id;
		
		ofstream fp; 
		fp.open(argv[3]);
		if(!fp){cout<<"Fail to open file: "<<filename<<endl; } 
		
		while(getline(file2,buffer)){
			buffer2 = buffer.substr(1);
			tmp = buffer2.substr( buffer2.find("\\")+1);
			tmp = tmp.substr( tmp.find("_")+1);
			tmp = tmp.substr(0, tmp.size()-4);
			myStr2Int(tmp, id);
					
			buffer = "ImageData/Flickr"+buffer;
			for (int i=0; i<buffer.size(); i++){
				if (buffer[i]==92) buffer[i]=47;
			}
			strcpy(buffer_c,buffer.c_str());
			filename = buffer_c; 
			Mat img=imread(filename,CV_LOAD_IMAGE_GRAYSCALE);		
			vector<KeyPoint> keypoints;		
			detector->detect(img,keypoints);
			Mat bowDescriptor;
			bowDE.compute(img,keypoints,bowDescriptor);

			//prepare the yml (some what similar to xml) file
			vector<float> V; 
			V.assign((float*)bowDescriptor.datastart, (float*)bowDescriptor.dataend);
			cout << "Vector: " <<V.size()<<endl; 
			fp<<tmp<<" ";
			for(int i=0;i<V.size();++i) { 
				fp<<V[i]<<" ";
				//cout<<V[i]<<" ";
			} cout<<endl; fp<<endl;
			
			//sprintf(imageTag,"img %d", count);			
			//fs1 << imageTag << bowDescriptor;
			count++;
		}
		
				

		//You may use this descriptor for classifying the image.

		//fs1.release();
		fp.close();
		printf("\ndone\n");	
	
	}
	
	/*******************************************************************
 							BOW EUCLIDEAN DISTANCE
 	********************************************************************/
 	
	else if (argv[1][1]=='4') {
		fstream file1(argv[2]); //train_bow.txt
		fstream file2(argv[3]); //test_bow.txt
		int num_train, num_test;
		vector<double> eu_train, test_now, train_now;
		vector< vector<double> > eu_test, train_t;
		string buffer, buffer2;
			
		while(getline(file1,buffer2)){
				char buffer2_c[1000];
				char *pch2;
				strcpy(buffer2_c, buffer2.c_str());
				pch2 = strtok(buffer2_c, " ");
			
				while(pch2!=NULL){
					double tmp = atof(pch2);
					train_now.push_back(tmp);
					pch2 = strtok(NULL," ");
				}
				
				if(train_now.size()!=500) continue;
				train_t.push_back(train_now);
				train_now.clear();	
				buffer2.clear();			
		}
		cout<<"Number of training data: "<<train_t.size()<<endl;
		
		
		for (int i=0; i<1; i++){
			getline(file2,buffer);
			char buffer_c[1000];
			char *pch;
			strcpy(buffer_c, buffer.c_str());
			pch = strtok(buffer_c, " ");
			while(pch!=NULL){
				double tmp = atof(pch);
				test_now.push_back(tmp);
				pch = strtok(NULL," ");
			}
			
			if(test_now.size()==500) cout<<"conti..."<<endl;
			else {cout<<"error: "<<test_now.size()<<endl;; return 0;}

			for (int j=0; j<train_t.size(); j++){
				double eu_tmp=0.0;
				for (int k=0; k<500; k++){
					eu_tmp = eu_tmp+pow((test_now[k]-train_t[j][k]),2);
				}
				
				eu_train.push_back(eu_tmp);		
			
			}
			cout<<endl;
			eu_test.push_back(eu_train);
			eu_train.clear();
			test_now.clear();
		}
		cout<<"done"<<endl;
		
		
		
	}
		
	
    return 0;
}
