#include <stdio.h>
#include <string.h>

template <typename T>
void DrawHists(const char* name, TTree* tree, int nBins){  
  cout<<name<<"   and   "<<tree->GetLeaf(name)->GetTypeName()<< "   and  " << '\n';
  T val;
  Char_t p_TruthType;
  
  tree->SetBranchAddress(name, &val);
  tree->SetBranchAddress("p_TruthType", &p_TruthType);
  
  //tree->SetBranchStatus("*",0);
  //tree->SetBranchStatus(name,1);
  //tree->SetBranchStatus("p_TruthType",1);

  
  std::string tempname =+ name;
  tempname += ">>his";
  const char *foo = tempname.c_str();  
  tree->Draw(foo,"","goff");
  
  T Max=tree->GetMaximum(name);
  T Min=tree->GetMinimum(name);
  
  // drawing some histograms
  TH1F *hist_all = new TH1F("hist_all", "", nBins, Min, Max); //TH1F is a 1-D histogram with one float per channel it takes "name", "title"
  TH1F *hist_sig = new TH1F("hist_sig", "", nBins, Min, Max); // bin size, lower bound, upper bound. 
  TH1F *hist_bkg = new TH1F("hist_bkg", "", nBins, Min, Max);

  for ( Int_t iev = 0; iev < tree->GetEntries(); iev++ ) { //get entries gets the number of entries in the tree 
    
    tree->GetEntry(iev);
      
    hist_all->Fill(val); //the hist with all the events

    if ( p_TruthType == 2 ) {   // only signal events
      hist_sig->Fill(val);
    }
    else {    // only background events
      hist_bkg->Fill(val);
    } 	
  }
  char print_name[50];
  int retVal=sprintf(print_name, "../graphs/%s_plot.gif",name);
  
  TCanvas *c1 = new TCanvas("c1", "c1", 800, 800);
  c1->SetLogy(1);   // set log scale on y axis
  
  hist_all->GetXaxis()->SetTitle(name);
  hist_sig->SetFillColor(kBlue-9);
  hist_sig->SetLineWidth(0);
  hist_bkg->SetMarkerStyle(21);
  
  TLegend *legend = new TLegend(0.25, 0.85, 0.55, 0.60); //defining the legend
  legend->AddEntry(hist_all, "all events", "l");
  legend->AddEntry(hist_sig, "only signal", "f");
  legend->AddEntry(hist_bkg, "only background", "p");
  legend->SetFillColor(kWhite);
  legend->SetBorderSize(0);
  
  
  hist_all->Draw("hist"); //draws everything in the same hist
  hist_sig->Draw("histsame");
  hist_bkg->Draw("esame");
  legend->Draw();
  
  c1->SaveAs(print_name);
  return 0;
}


void hists_copy() {
  // delare histograms
  
  int nBins = 50; //changed number of ins from 50 to 40, am i free to do so?
  
 
  // open input root file and tree

  TFile *inFile = new TFile ("../forward_MC/user.lehrke.mc16_13TeV.361106.Zee.EGAM8.e3601_e5984_s3126_r10724_r10726_p3648.ePID18_NTUP3_v01_myOutput.root/user.lehrke.17118381._000003.myOutput.root");
  TTree *tree = (TTree*)inFile->Get("tree");
  TObjArray *names =(TObjArray*)tree->GetListOfLeaves()->Clone(); //creates an object from which names can be grabbed  
  for (int i = 0; i < names->GetEntries(); ++i){
      //TObject *name//
    const char* name=(const char*)names->operator[](i)->GetName();
    
    if (strncmp (tree->GetLeaf(name)->GetTypeName(),"Float_t",5) == 0){
      //float name_double = strtod(name, NULL);
      //cout<<name_double<< '\n';
      DrawHists<float>(name,tree,nBins);
    }
    else if (strncmp (tree->GetLeaf(name)->GetTypeName(),"Int_t",5) == 0 ){
      //Int_t name_double = strtod(name, NULL);
      //cout<<name_double<< '\n';
      DrawHists<Int_t>(name,tree,nBins);
    }
    else if (strncmp (tree->GetLeaf(name)->GetTypeName(),"string",5) == 0 ){
      //Int_t name_double = strtod(name, NULL);
      //cout<<name_double<< '\n';
    }
    else {
      //Char_t name_double = strtod(name, NULL);
      //cout<<"here" <<name_double<< "here"<< '\n';
      DrawHists<Char_t>(name,tree,nBins);
    } 
  }
}
