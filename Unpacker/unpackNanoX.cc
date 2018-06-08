#include "TFile.h"
#include "TTree.h"
#include "TTreeFormula.h"

#include <iostream>
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>

class NanoXTree
{
    private:
        TTree* tree_;
        std::vector<std::shared_ptr<TTreeFormula>> expr_;
        std::vector<std::shared_ptr<TTreeFormula>> multiplicity_;
        std::vector<std::string> names_;
        int ientry_;
    public:
        NanoXTree(TTree* tree):
            tree_(tree),
            ientry_(-1)
        {
        }
        
        void addExpr(
            const std::string& name, 
            const std::string& expr, 
            const std::string& multiplicity = "")
        {
            expr_.emplace_back(
                std::make_shared<TTreeFormula>(name.c_str(),expr.c_str(),tree_)
            );
            if (multiplicity.size()==0)
            {
                multiplicity_.emplace_back(nullptr);
            }
            else
            {
                multiplicity_.emplace_back(
                    std::make_shared<TTreeFormula>((name+multiplicity).c_str(),multiplicity.c_str(),tree_)
                );
            }
            names_.push_back(name);
        }
        
        inline void addExpr(const std::string& name)
        {
            addExpr(name,name);
        }
        
        inline int entries() const
        {
            return tree_->GetEntries();
        }
        
        int getEvent(int entry)
        {
            if (entry!=ientry_)
            {
                tree_->GetEntry(entry);
                ientry_ = entry;
                for (unsigned int i = 0; i < names_.size(); ++i)
                {
                    if (multiplicity_[i])
                    {
                        multiplicity_[i]->GetNdata();
                    }
                }
            }
            for (unsigned int i = 0; i < names_.size(); ++i)
            {
                if (not multiplicity_[i])
                {
                    return expr_[i]->GetNdata();
                }
            }
            return 0;
        }
        
        int getJet(
            int entry, 
            int jet, 
            std::unordered_map<std::string,float>& single,
            std::unordered_map<std::string,std::vector<float>>& multi
        )
        {
            int njets = getEvent(entry);
            if (njets<jet)
            {
                return 1;
            }
            for (unsigned int i = 0; i < names_.size(); ++i)
            {
                if (multiplicity_[i])
                {
                    int nentries = multiplicity_[i]->EvalInstance(jet);
                    int offset = 0;
                    for (unsigned int ioff = 0; ioff < jet; ++ioff)
                    {
                        offset += multiplicity_[i]->EvalInstance(ioff);
                    }
                    multi[names_[i]] = std::vector<float>(nentries,0);
                    for (unsigned int m = 0; m < nentries; ++m)
                    {
                        multi[names_[i]][m] = expr_[i]->EvalInstance(offset+m);
                    }
                }
                else
                {
                    single[names_[i]] = expr_[i]->EvalInstance(jet);
                }
            }
        }
}; 

void printSyntax()
{
    std::cout<<"Syntax: "<<std::endl;
    std::cout<<"          unpackNanoX noutputs infile [infile [infile ...]]"<<std::endl<<std::endl;
}

int main(int argc, char **argv)
{
    if (argc<3)
    {
        printSyntax();
        return 1;
    }
    int nOutputs = std::atoi(argv[1]);
    if (nOutputs==0 and strcmp(argv[1],"0")!=0)
    {
        std::cout<<"Error - cannot convert '"<<argv[1]<<"' to integer"<<std::endl;
        return 1;
    }
    else if (nOutputs<=0)
    {
        std::cout<<"Error - noutputs need to be positive but got '"<<argv[1]<<"'"<<std::endl;
        return 1;
    }
    
    
    
    std::vector<std::shared_ptr<TFile>> files;
    std::vector<TTree*> trees;
    std::cout<<"Input files: "<<std::endl;
    for (unsigned int iarg = 2; iarg<argc; ++iarg)
    {
        //std::cout<<"   "<<argv[iarg]<<", nEvents="<<;
        TFile* file = TFile::Open(argv[iarg]);
        if (not file)
        {
            std::cout<<"File '"<<argv[iarg]<<"' cannot be read"<<std::endl;
            continue;
        }
        
        TTree* tree = dynamic_cast<TTree*>(file->Get("Events"));
        
        if (not tree)
        {
            std::cout<<"Tree in file '"<<argv[iarg]<<"' cannot be read"<<std::endl;
            continue;
        }
        std::cout<<"   "<<argv[iarg]<<", nEvents="<<tree->GetEntries()<<std::endl;
    }
    NanoXTree nano(trees.back());
    
    return 0;
}
