/*===================================================================
Copyright 2019 Matthias Komm, Vilius Cepaitis, Robert Bainbridge, 
Alex Tapper, Oliver Buchmueller. All Rights Reserved. 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
    
Unless required by applicable law or agreed to in writing, 
software distributed under the License is distributed on an "AS IS" 
BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express 
or implied.See the License for the specific language governing 
permissions and limitations under the License.
===================================================================*/


#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/op.h"

using namespace tensorflow;

namespace syntax_test
{
    static bool isArray(const string& s)
    {
        auto p1 = std::find(s.begin(),s.end(),'[');
        auto p2 = std::find(s.begin(),s.end(),']');
        return p1!=s.end() and p2!=s.end() and p1<p2;
    }
}

REGISTER_OP("RootReader")
    .Input("queue_handle: resource")
    .Attr("branches: list(string)")
    .Attr("treename: string")
    .Attr("naninf: float = 0")
    .Attr("batch: int = 1")
    .Output("out: float32")
    .Output("num: int32")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
        std::vector<string> branchNames;
        TF_RETURN_IF_ERROR(c->GetAttr("branches",&branchNames));
        int size = 0;
        for (auto name: branchNames)
        {
            
            if (not syntax_test::isArray(name))
            {
                size+=1;
            }
            else
            {
                auto p1 = std::find(name.begin(),name.end(),'[');
                auto p2 = std::find(p1,name.end(),']');
                
                size += std::stoi(std::string(p1+1,p2));
            }
        }
        shape_inference::ShapeHandle s1 = c->MakeShape({-1,c->MakeDim(size)});
        c->set_output(0, s1);
        
        shape_inference::ShapeHandle s2 = c->MakeShape({-1,1});
        c->set_output(1, s2);
        return Status::OK();
    });

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "RootMutex.h"

#include "TFile.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TTreeFormula.h"

#include <vector>
#include <memory>

#include <chrono>
#include <thread>



class RootReaderOp:
    public OpKernel
{

    public:
        template<typename OUT>
        class TensorFiller
        {
            protected:
                string name_;
                string expr_;
            public:
                TensorFiller(const string& name, const string& expr):
                    name_(name),
                    expr_(expr)
                {
                }
                
                inline const string name() const
                {
                    return name_;
                }
                
                inline const string expr() const
                {
                    return expr_;
                }
                
                static OUT resetNanOrInf(const OUT& v, const OUT& reset)
                {
                    if (std::isnan(v) or std::isinf(v))
                    {
                        return reset;
                    }
                    return v;
                }
                
                virtual void setBranchAddress(OpKernelContext* context, TTree* tree) = 0;
                virtual int fillTensor(
                    typename TTypes<OUT>::Flat& flatTensor,
                    int index,
                    const OUT& reset
                ) const = 0;
                virtual ~TensorFiller() {}
        
        };
        
        
        template<typename IN, typename OUT=IN>
        class TensorFillerTmpl:
            public TensorFiller<OUT>
        {
            private:
                int size_;
                std::unique_ptr<TTreeFormula> formula_;
            public:
                typedef TensorFiller<OUT> Base;
            
                TensorFillerTmpl(
                    OpKernelConstruction*, 
                    const string& name, 
                    const string& expr, 
                    int size
                ):
                    TensorFiller<OUT>(name,expr),
                    size_(size),
                    formula_(nullptr)
                {
                }
                
                virtual ~TensorFillerTmpl()
                {
                }
                
                virtual void setBranchAddress(OpKernelContext* context, TTree* tree)
                {
                    std::size_t id = std::hash<std::thread::id>()(std::this_thread::get_id());
                    formula_.reset(new TTreeFormula(
                        (Base::name()+std::to_string(id)).c_str(),
                        Base::expr().c_str(),
                        tree
                    ));
                    
                    if(not formula_ or formula_->GetNdim()==0)
                    {
                        context->CtxFailureWithWarning(__FILE__, __LINE__, Status(error::INVALID_ARGUMENT,"Cannot parse equation '"+Base::expr()+"'"));
                        return;
                    }
                    formula_->SetQuickLoad(true);
                }
                
                virtual int fillTensor(
                    typename TTypes<OUT>::Flat& flatTensor, 
                    int index, 
                    const OUT& reset
                ) const
                {
                    //needs to be called; otherwise elements >0 are set to 0
                    int leafSize = formula_->GetNdata(); 
                    for (int i = 0; i < std::min<int>(leafSize,size_); ++i)
                    {
                        OUT result = static_cast<OUT>(formula_->EvalInstance(i));
                        flatTensor(index+i)=Base::resetNanOrInf(result,reset);
                    }
                    for (int i = std::min(leafSize,size_); i < size_; ++i)
                    {
                        flatTensor(index+i) = reset; //padding
                    }
                    return index+size_;
                }
        };
        
    private:
        mutex localMutex_; //protects class members
        std::unique_ptr<TFile> inputFile_;
        TTree* tree_;
        std::vector<std::unique_ptr<TensorFiller<float>>> tensorFillers_;
        int currentEntry_;
        
        float naninf_;
        string treename_;
        int size_;
        int nBatch_;
        int nEvents_;
        
    public:
        explicit RootReaderOp(OpKernelConstruction* context): 
            OpKernel(context),
            inputFile_(nullptr),
            currentEntry_(0),
            naninf_(0),
            size_(0),
            nBatch_(1),
            nEvents_(0)
        {
            RootMutex::Lock lock = RootMutex::lock();
            
            std::vector<string> branchNames;
            OP_REQUIRES_OK(
                context,
                context->GetAttr("branches",&branchNames)
            );
            OP_REQUIRES_OK(
                context,
                context->GetAttr("treename",&treename_)
            );
            OP_REQUIRES_OK(
                context,
                context->GetAttr("naninf",&naninf_)
            );
            OP_REQUIRES_OK(
                context,
                context->GetAttr("batch",&nBatch_)
            );
            for (unsigned int iname = 0; iname < branchNames.size(); ++iname)
            {
                const string& name = branchNames[iname];
                if (not syntax_test::isArray(name))
                {
                    auto it = std::find(name.begin(),name.end(),'/');
                    if (it==name.end())
                    {
                        tensorFillers_.emplace_back(
                            std::make_unique<
                                TensorFillerTmpl<float>
                            >(
                                context,
                                "expr_"+std::to_string(iname),
                                name,
                                1
                            )
                        );
                        size_+=1;
                    }
                }
                else
                {
                    auto p1 = std::find(name.begin(),name.end(),'[');
                    auto p2 = std::find(p1,name.end(),']');
                    std::string branchName(name.begin(),p1);
                    int size = std::stoi(std::string(p1+1,p2));
                    size_+=size;
                    tensorFillers_.emplace_back(
                        std::make_unique<TensorFillerTmpl<float>>(
                            context,
                            "expr_"+std::to_string(iname),
                            branchName,
                            size
                        )
                    );
                }
            }
        }
        
        virtual ~RootReaderOp()
        {
            RootMutex::Lock lock = RootMutex::lock();
            tensorFillers_.clear();
            if (inputFile_) inputFile_->Close();
        }
        
        void Compute(OpKernelContext* context)
        {
            mutex_lock localLock(localMutex_);
           
            while (not inputFile_)
            {
                QueueInterface* queue;
                OP_REQUIRES_OK(context,GetResourceFromContext(
                    context, "queue_handle", &queue
                ));
                
                string fileName = GetNextFilename(queue,context);
                if (!context->status().ok())
                {
                    return;
                }
                if (fileName.size()==0) 
                {
                    context->CtxFailureWithWarning(__FILE__, __LINE__, Status(error::INVALID_ARGUMENT,"Got empty filename"));
                    return;
                }
                RootMutex::Lock lock = RootMutex::lock();
                TFile* f = TFile::Open(fileName.c_str());
                if (not (f and f->IsOpen()))
                {
                    context->CtxFailureWithWarning(__FILE__, __LINE__, Status(error::INVALID_ARGUMENT,"Cannot read file '"+fileName+"'"));
                    return;
                }
                inputFile_.reset(f);
                
                currentEntry_ = 0;
                tree_ = dynamic_cast<TTree*>(inputFile_->Get(treename_.c_str()));
                if (not tree_)
                {
                    context->CtxFailureWithWarning(__FILE__, __LINE__, Status(error::INVALID_ARGUMENT,"Cannot get tree '"+treename_+"' from file '"+fileName+"'"));
                    return;
                }
                for (auto& tensorFiller: tensorFillers_)
                {
                    tensorFiller->setBranchAddress(context,tree_);
                    if (not context->status().ok()) return;
                }
                nEvents_ = static_cast<int>(tree_->GetEntries());
            }
            Tensor* output_tensor = nullptr;
            TensorShape shape;
            int64_t nBatches = std::min<int>(nEvents_-currentEntry_,nBatch_);
            shape.AddDim(nBatches);
            shape.AddDim(size_);
            OP_REQUIRES_OK(context, context->allocate_output("out", shape,&output_tensor));
            
            Tensor* output_num = nullptr;
            TensorShape shape_num;
            shape_num.AddDim(nBatches);
            shape_num.AddDim(1);
            OP_REQUIRES_OK(context, context->allocate_output("num", shape_num,&output_num));
            
            auto output_flat = output_tensor->flat<float>();
            auto output_num_flat = output_num->flat<int>();
            int index = 0;
            
            for (int64_t ibatch=0; ibatch<nBatches;++ibatch)
            {
                tree_->GetEntry(currentEntry_);
                output_num_flat(ibatch)=currentEntry_;
                for (auto& tensorFiller: tensorFillers_)
                {
                    index = tensorFiller->fillTensor(output_flat,index,naninf_);
                }
                ++currentEntry_;
            }
            if (currentEntry_>=nEvents_)
            {
                RootMutex::Lock lock = RootMutex::lock();
                inputFile_.reset(nullptr);
            }
        }
        
        string GetNextFilename(QueueInterface* queue, OpKernelContext* context) const 
        {
            string work;
            Notification n;
            queue->TryDequeue(
                context, [this, context, &n, &work](const QueueInterface::Tuple& tuple) 
                {
                    if (context->status().ok())
                    {
                        if (tuple.size() != 1) 
                        {
                            context->CtxFailureWithWarning(__FILE__, __LINE__, Status(error::INVALID_ARGUMENT,
                                "Expected single component queue"
                            ));
                        } 
                        else if (tuple[0].dtype() != DT_STRING) 
                        {
                            context->CtxFailureWithWarning(__FILE__, __LINE__, Status(error::INVALID_ARGUMENT,
                                "Expected queue with single string component"
                            ));
                        } 
                        else if (tuple[0].NumElements() != 1) 
                        {
                            context->CtxFailureWithWarning(__FILE__, __LINE__, Status(error::INVALID_ARGUMENT,
                                "Expected to dequeue a one-element string tensor"
                            ));
                        } 
                        else 
                        {
                            work = tuple[0].flat<string>()(0);
                        }
                    }
                    n.Notify();
                }
            );
            n.WaitForNotification();
            return work;
        }   
};

REGISTER_KERNEL_BUILDER(Name("RootReader").Device(DEVICE_CPU),RootReaderOp)

