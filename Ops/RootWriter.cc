#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/op.h"



using namespace tensorflow;


REGISTER_OP("RootWriter")
    .Input("input: float32")
    .Input("write: int32")
    .Attr("branches: list(string)")
    .Attr("treename: string")
    .Attr("filename: string")
    //.Output("output: int32")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
        //shape_inference::ShapeHandle s = c->MakeShape({1});
        //c->set_output(0,s);
        return Status::OK();
    })
    .Doc(R"doc(A Reader that outputs the lines of a file delimited by '\n'.)doc");

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "RootMutex.h"

#include "TFile.h"
#include "TTree.h"

#include <vector>
#include <memory>

#include <chrono>
#include <thread>

class RootWriterOp:
    public OpKernel
{

    public:
        template<typename OUT>
        class Branch
        {
            protected:
                string name_;
            public:
                Branch(const string& name):
                    name_(name)
                {
                }
                
                inline const string name() const
                {
                    return name_;
                }
                
                static OUT resetNanOrInf(const OUT& v, const OUT& reset)
                {
                    if (std::isnan(v) or std::isinf(v))
                    {
                        return reset;
                    }
                    return v;
                }
                
                virtual void setBranchAddress(TTree* tree) = 0;
                virtual void bookBranchAddress(TTree* tree) = 0;
                virtual unsigned int fillTensor(typename TTypes<OUT>::Flat& flatTensor, unsigned int index, const OUT& reset) const = 0;
        };
        
        template<typename IN, typename OUT=IN>
        class SingleBranch:
            public Branch<OUT>
        {
            private:
                IN value_;
            public:
                SingleBranch(const string& name):
                    Branch<OUT>(name)
                {
                }
                
                inline const IN& value() const
                {
                    return value_;
                }
                
                inline IN& value()
                {
                    return value_;
                }
                
                virtual void bookBranchAddress(TTree* tree)
                {
                    tree->Branch(Branch<OUT>::name().c_str(),&value_);
                    //std::cout<<"booking branch "<<Branch<OUT>::name()<<std::endl;
                }
                
                virtual void setBranchAddress(TTree* tree)
                {
                    if(tree->SetBranchAddress(Branch<OUT>::name().c_str(),&value_)<0)
                    {
                        throw std::runtime_error("No branch with name '"+Branch<OUT>::name()+"' in tree");
                    }
                }
                
                virtual unsigned int fillTensor(typename TTypes<OUT>::Flat& flatTensor,unsigned int index, const OUT& reset) const
                {
                    flatTensor(index)=Branch<OUT>::resetNanOrInf(value_,reset);
                    //std::cout<<index<<": "<<Branch<OUT>::name()<<"="<<flatTensor(index)<<std::endl;
                    return index+1;
                }
        }; 
        
        template<typename IN, typename OUT=IN>
        class ArrayBranch:
            public Branch<OUT>
        {
            private:
                alignas(16) IN values_[50]; //there is some odd bug when using dynamic allocated arrays and root
                unsigned int size_;
                std::shared_ptr<SingleBranch<unsigned int,OUT>> length_;
                 
            public:
                ArrayBranch(const string& name, std::shared_ptr<SingleBranch<unsigned int,OUT>>& length, unsigned int size):
                    Branch<OUT>(name),
                    length_(length),
                    size_(size)
                {
                }
                
                inline const IN& value(unsigned int index) const
                {
                    if (index>=size_)
                    {
                        throw std::runtime_error("Array index out-of-range");
                    }
                    return values_[index];
                }
                
                virtual ~ArrayBranch()
                {
                    //delete[] values_;
                }
                
                virtual void bookBranchAddress(TTree* tree)
                {
                    tree->Branch(Branch<OUT>::name().c_str(),&values_);
                }
                
                //error codes: https://root.cern.ch/doc/master/classTTree.html#a1a48bf75621868a514741b27252cad96
                virtual void setBranchAddress(TTree* tree)
                {
                    if(tree->SetBranchAddress(Branch<OUT>::name().c_str(),values_)<0)
                    {
                        throw std::runtime_error("No branch with name '"+Branch<OUT>::name()+"' in tree");
                    }
                }
                virtual unsigned int fillTensor(typename TTypes<OUT>::Flat& flatTensor, unsigned int index, const OUT& reset) const
                {
                    //std::cout<<Branch<OUT>::name()<<", length="<<length_->value()<<std::endl;
                    for (unsigned int i = 0; i < std::min(length_->value(),size_); ++i)
                    {
                        //std::cout<<(index+i)<<": "<<values_[i]<<std::endl;
                        flatTensor(index+i)=Branch<OUT>::resetNanOrInf(values_[i],reset);
                    }
                    for (unsigned int i = std::min(length_->value(),size_); i < size_; ++i)
                    {
                        //std::cout<<(index+i)<<": padded"<<std::endl;
                        flatTensor(index+i) = 0; //zero padding
                    }
                    
                    return index+size_;
                }
        };
        
    private:
        mutex localMutex_; //protects class members
        std::unique_ptr<TFile> outputFile_;
        TTree* tree_;
        std::vector<std::shared_ptr<SingleBranch<float>>> branches_;
        
    public:
        explicit RootWriterOp(OpKernelConstruction* context): 
            OpKernel(context),
            outputFile_(nullptr)
        {
            
            std::vector<string> branchNames;
            OP_REQUIRES_OK(
                context,
                context->GetAttr("branches",&branchNames)
            );
            string tree_name;
            OP_REQUIRES_OK(
                context,
                context->GetAttr("treename",&tree_name)
            );
            string file_name;
            OP_REQUIRES_OK(
                context,
                context->GetAttr("filename",&file_name)
            );
            
            RootMutex::Lock lock = RootMutex::lock();
            
            outputFile_.reset(new TFile(file_name.c_str(),"RECREATE"));
            tree_ = new TTree(tree_name.c_str(),tree_name.c_str());
            tree_->SetDirectory(outputFile_.get());
            
            for (auto& name: branchNames)
            {
                auto p = std::find(name.begin(),name.end(),'/');
                string branchName = name;
                if (p!=name.end())
                {
                    branchName = string(name.begin(),p);
                }   
                //std::cout<<name<<" = default"<<std::endl;
                std::shared_ptr<SingleBranch<float>> singleBranch = std::make_shared<SingleBranch<float>>(branchName);
                singleBranch->bookBranchAddress(tree_);
                branches_.emplace_back(singleBranch);
            }
            
        }
        
        virtual ~RootWriterOp()
        {
        }

        void Compute(OpKernelContext* context)
        {
            //std::cout<<"compute writer"<<std::endl;
            
            mutex_lock localLock(localMutex_);
            
            if (not outputFile_)
            {
                throw std::runtime_error("Output file not opened");
            }
            
            const Tensor& input_tensor = context->input(0);
            auto input = input_tensor.flat<float>();
            int num_batches = input_tensor.dim_size(0);
            int label_length = input_tensor.dim_size(1);
            
            if (label_length!=branches_.size())
            {
                throw std::runtime_error("Mismatching tensor ("+std::to_string(label_length)+")  <-> branch length ("+std::to_string(branches_.size())+")");
            }
            for (unsigned int ibatch = 0; ibatch < num_batches; ++ibatch)
            {
                for (unsigned int i = 0; i < label_length; ++i)
                {
                    //std::cout<<"writing "<<branches_[i]->name()<<" = "<<input(i)<<std::endl;
                    branches_[i]->value()=input(ibatch*label_length+i);
                }
                RootMutex::Lock lock = RootMutex::lock();
                outputFile_->cd();
                tree_->Fill();
            }
            
            const Tensor& write_tensor = context->input(1);
            auto write_flag = write_tensor.flat<int>();
            if (write_flag(0)==0)
            {
                std::cout<<"closing file"<<std::endl;
                //outputFile_->cd();
                //TFile* file = tree_->GetCurrentFile();
                std::cout<<"write tree: "<<bool(tree_)<<std::endl;
                std::cout<<"entries: "<<tree_->GetEntries()<<std::endl;
                tree_->Write();
                std::cout<<"close file: "<<bool(outputFile_)<<std::endl;
                outputFile_->Close();
                std::cout<<"reset file"<<std::endl;
                branches_.clear();
                outputFile_.reset();
            }
        }
            
};


REGISTER_KERNEL_BUILDER(Name("RootWriter").Device(DEVICE_CPU),RootWriterOp);

