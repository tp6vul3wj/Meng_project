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
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Eigen>
#include <Eigen/Dense>
using namespace cv;
using namespace std;
using Eigen::MatrixXd;
using Eigen::EigenSolver;
/**************************************************************************************************

./bin/bow data/LMO/LMO_gist.txt data/LMO/LMO_anchor.txt data/LMO/LMO_test.txt data/LMO/LMO_train.txt

**************************************************************************************************/


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

bool readfile_vector_int(fstream &file, vector<int> &out_vec)
{
	string buffer;
	while(getline(file,buffer)){
		int tmp;
		myStr2Int(buffer, tmp);
		out_vec.push_back(tmp);
	}
	
	return true;
}

bool readfile_vector_str(fstream &file, vector<string> &out_vec)
{
	string buffer;
	while(getline(file,buffer)){
		out_vec.push_back(buffer);
	}
	
	return true;
}


bool readfile_matrix(fstream &file, vector< vector<double> > &out_max)
{
	string buffer;
	vector<double> single_line;
	while(getline(file,buffer)){
		char buffer_c[5000];
		char *pch;
		strcpy(buffer_c, buffer.c_str());
		pch = strtok(buffer_c, " ");
		while(pch!=NULL){
			double tmp = atof(pch);
			single_line.push_back(tmp);
			pch = strtok(NULL," ");
		}
		
		if(single_line.size()!=512) cout<<"error: "<<single_line.size()<<endl;
		out_max.push_back(single_line);
		single_line.clear();
	}
	
}


bool sqdist(vector< vector<double> > &a, vector< vector<double> > &b, vector< vector<double> > &d)
{

	vector<double> aa; //aa = sum(a.*a, 1)
	double ele_aa;
	for (int i=0; i<a.size(); i++){
		ele_aa = 0.0;
		for (int j=0; j<a[0].size(); j++){
			ele_aa = ele_aa + a[i][j]*a[i][j];
		}
		aa.push_back(ele_aa);
	}
	
	/*
	cout<<"[aa]"<<endl;
	cout<<"size of aa = "<<aa.size()<<endl;
	for (int i=0; i<aa.size(); i++) cout<<aa[i]<<" ";
	cout<<endl;
	*/

	vector<double> bb; //aa = sum(a.*a, 1)
	double ele_bb;
	for (int i=0; i<b.size(); i++){
		ele_bb = 0.0;
		for (int j=0; j<b[0].size(); j++){
			ele_bb = ele_bb + b[i][j]*b[i][j];
		}
		bb.push_back(ele_bb);
	}

	
	vector< vector<double> > ab; // ab = a'*b;
	vector<double> sub_ab;
	double ele_ab;
	for (int i=0; i<a.size(); i++){ // #training data = 2500 
		for (int j=0; j<b.size(); j++){ // #anchor size = 300
			ele_ab = 0.0;
			for (int k=0; k<a[0].size(); k++){ // dim of feature = 512
				
				ele_ab = ele_ab + a[i][k]*b[j][k];
			}
			sub_ab.push_back(ele_ab);
		}
		
		if (sub_ab.size()!=300) {cout<<"error"<<endl; return 0;}
		ab.push_back(sub_ab);
		sub_ab.clear();
	}
	
	/*
	cout<<"Size of ab = "<<ab.size()<<"*"<<ab[0].size()<<endl;
	for(int i=0; i<300; i++) cout<<ab[0][i]<<" ";
	cout<<endl<<endl;
	*/
	
	vector<double> sub_d;
	double ele_d;
	for (int i=0; i<ab.size(); i++){ //2500
		for (int j=0; j<ab[0].size(); j++){
			ele_d = abs(aa[i]+bb[j]-2*ab[i][j]);
			sub_d.push_back(ele_d);
		}
		d.push_back(sub_d);
		sub_d.clear();
	}

	return 1;
}

bool sort_pos(vector<double> &v, int &pos, double &val){
	vector<double> v2 = v;
	std::sort(v2.begin(), v2.end());
	val = v2[0];
	for (int i=0; i<v.size(); i++){
		if (v[i]==val){
			pos = i+1;  // pos = position+1 !!
			break;
		}
	}
}

bool OneLayerAGH_Train(vector< vector<double> > &train, vector< vector<double> > &anchor, int r, int s, double sigma, vector< vector<bool> > &Y, vector< vector<double> > &W )
{
	
	int n = train.size(); //2500
	int m = anchor.size(); //300
	vector< vector<double> > d;
	sqdist(train, anchor, d);
	//cout<<"Size of d = "<<d.size()<<"*"<<d[0].size()<<endl;
	//for(int i=0; i<300; i++) cout<<d[0][i]<<" ";
	vector< vector<double> > val, pos;
	vector<double> sub_val, sub_pos;
	int ele_pos;
	double ele_val;
	for(int i=0; i<s; i++){
		for(int j=0; j<d.size(); j++){ //2500
			sort_pos(d[j], ele_pos, ele_val);
			//cout<<"d["<<j<<"] = ("<<ele_pos<<","<<ele_val<<")"<<" ";
			sub_pos.push_back(ele_pos);
			sub_val.push_back(ele_val);
			d[j][ele_pos-1] = 1000000;
		}
		val.push_back(sub_val);
		pos.push_back(sub_pos);
		sub_pos.clear();
		sub_val.clear();
	}
	
	if (sigma==0){
		double tmp_sig=0.0;
		for (int i=0; i<val[0].size(); i++){
			tmp_sig = tmp_sig+sqrt(val[s-1][i]);
		}
		sigma = tmp_sig/val[0].size();
	}
	cout<<"sigma = "<<sigma<<endl;
	/*
	cout<<"val size = "<<val.size()<<"*"<<val[0].size()<<endl;
	cout<<"pos size = "<<val.size()<<"*"<<pos[0].size()<<endl;
	cout<<val[0][100]<<" "<<val[1][100]<<" "<<val[0][2499]<<" "<<val[1][2499]<<endl;
	cout<<pos[0][100]<<" "<<pos[1][100]<<" "<<pos[0][2499]<<" "<<pos[1][2499]<<endl;
	*/
	
	for (int i=0; i<val.size(); i++){
		for (int j=0; j<val[0].size(); j++){
			val[i][j] = exp(-val[i][j]/(sigma*sigma));
			//cout<<val[i][j]<<" ";
		}
		//cout<<endl<<endl;
	}

	for (int i=0; i<val[0].size(); i++){
		double val1, val2;
		val1 = val[0][i];
		val2 = val[1][i];
		val[0][i] = val1/(val1+val2);
		val[1][i] = val2/(val1+val2);
		//cout<<"("<<val[0][i]<<","<<val[1][i]<<") ";
	}
	
	// Z
	vector< vector<double> > Z;
	vector<double> sub_Z;
	for (int i=0 ;i<m; i++) sub_Z.push_back(0);
	for (int i=0; i<n; i++) Z.push_back(sub_Z);
	cout<<"Z size = "<<Z.size()<<"*"<<Z[0].size()<<endl;
	
	for (int i=0; i<n; i++){
		for (int j=0; j<s; j++){
			int tmp_pos = pos[j][i]-1;
			Z[i][tmp_pos] = val[j][i];
			
		}
	}
	
	// lamda = sum(z) // 300=number of anchor
	vector<double> lamda;
	for (int i=0; i<m; i++) lamda.push_back(0.0);
	for (int i=0; i<m; i++){
		for (int j=0; j<n; j++){
			lamda[i] = lamda[i]+Z[j][i];
		}
	}

	/*
	cout<<"size of lamda is "<<lamda.size()<<endl;
	for (int i=0; i<m; i++) cout<<lamda[i]<<" ";
	cout<<endl;
	*/
	
	// M
	vector< vector<double> > M; 
	vector<double> sub_M;
	double ele_M;
	for (int i=0; i<m; i++){ 
		for (int j=0; j<m; j++){ 
			ele_M = 0.0;
			for (int k=0; k<n; k++){ 
				
				ele_M = ele_M + Z[k][i]*Z[k][j];
			}
			sub_M.push_back(ele_M);
		}
		
		if (sub_M.size()!=300) {cout<<"error"<<endl; return 0;}
		M.push_back(sub_M);
		sub_M.clear();
	}
	
	/*
	cout<<"Size of M = "<<M.size()<<"*"<<M[0].size()<<endl;
	for(int i=0; i<300; i++) cout<<M[0][i]<<" ";
	cout<<endl;
	for(int i=0; i<300; i++) cout<<M[299][i]<<" ";
	*/
	
	// lamda*M*lamda & convert M to matrxd
	MatrixXd M2(m,m);
	for (int i=0; i<M.size(); i++){
		for (int j=0; j<M[0].size(); j++){
			M[i][j] = M[i][j]*1/sqrt(lamda[i])*1/sqrt(lamda[j]);
			M2(i,j) = M[i][j];
		}
	
	}
	/*
	cout<<"Size of M = "<<M.size()<<"*"<<M[0].size()<<endl;
	for(int i=0; i<300; i++) cout<<M[5][i]<<" ";
	cout<<endl;
	for(int i=0; i<300; i++) cout<<M[299][i]<<" ";
	*/
	
	EigenSolver<MatrixXd> es(M2);
	vector<double> eigenvalue;
	for (int i=0; i<m; i++){
		eigenvalue.push_back(real(es.eigenvalues()[i]));
		//cout<<eigenvalue[i]<<" ";
	} //cout<<endl;
	
	
	vector<pair<double,size_t> > vp; 
	vp.reserve(eigenvalue.size()); 
	for (size_t i = 0 ; i != eigenvalue.size() ; i++) 
	{ vp.push_back(make_pair(eigenvalue[i], i)); } 
	sort(vp.begin(), vp.end()); 
	
	//for (size_t i = 0 ; i != vp.size() ; i++) { cout << vp[i].first << " " << vp[i].second << endl; }
	
	/*
	for (int i=0; i<m; i++){
		for (int j=0; j<m; j++){
			sub_W_tmp.push_back(real(es.eigenvectors()(i,j)));
			//cout<<real(es.eigenvectors()(i,j))<<" ";
		}//cout<<endl;
		W_tmp.push_back(sub_W_tmp);
		sub_W_tmp.clear();
	}
	*/
	
	MatrixXcd D = es.eigenvalues().asDiagonal();
	MatrixXcd V = es.eigenvectors();
	//es.eigenvectors()--> Eigen::Matrix<std::complex<double>
	//cout << "The eigenvalues of A are:" << endl << real(es.eigenvalues()[0]) << endl;
	//cout << "The matrix of eigenvectors, V, is:" << endl << es.eigenvectors() << endl << endl;
	
}



int main(int argc, char *argv[])
{	
	// read files
	fstream file_gist(argv[1]);
	fstream file_anchor(argv[2]);
	fstream file_test(argv[3]);
	fstream file_train(argv[4]);
	vector<int> train_idx, test_idx;
	vector< vector<double> > gist, anchor, train_gist;
	vector<string> namelist;
	
	readfile_vector_int(file_train, train_idx);
	cout<<"Number of testing data is "<<train_idx.size()<<endl;
	readfile_vector_int(file_test, test_idx);
	cout<<"Number of testing data is "<<test_idx.size()<<endl;
	
	readfile_matrix(file_anchor, anchor);
	cout<<"Size of dataset is "<<anchor.size()<<endl;
	cout<<"Dim of gist is "<<anchor[0].size()<<endl;
	
	readfile_matrix(file_gist, gist);
	cout<<"Size of dataset is "<<gist.size()<<endl;
	cout<<"Dim of gist is "<<gist[0].size()<<endl;
	
	// parameter setting
	int num_test = test_idx.size();
	int num_train = train_idx.size();
	int dim_feature = gist[0].size();
	int m = 300;
	int s = 2;
	int r = 12;
	double sigma = 0.0;
	
	
	vector<double> sub_train_gist;
	for(int i=0; i<num_train; i++){
		for (int j=0; j<dim_feature; j++){
			sub_train_gist.push_back(gist[train_idx[i]-1][j]);
		}
		train_gist.push_back(sub_train_gist);
		sub_train_gist.clear();
	}
	
	cout<<"Train_gist size :"<<train_gist.size()<<"*"<<train_gist[0].size()<<endl;

	
	vector< vector<bool> > Y;
	vector< vector<double> > W;
	 
	OneLayerAGH_Train(train_gist, anchor, r, s, sigma, Y, W);

	
	 MatrixXd mm(2,2);
 mm(0,0) = 3;
mm(1,0) = 2.5;
mm(0,1) = -1;
mm(1,1) = mm(1,0) + mm(0,1);
std::cout << mm << std::endl;
    return 0;
}
