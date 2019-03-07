#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/op.h"



using namespace tensorflow;

//NOTE: regex is experimental in gcc 4.9 and below
//TODO: define proper syntax and parsing e.g. support
//  <name>; <name>/<type>; <name>[<num>,<max>]; <name>[<num>,<max>]/<type>
//TODO (optional): add selections, add support for TSelector
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
    .Attr("naninf: int = 0")
    .Attr("batch: int = 1")
    .Output("out: float32")
    .Output("num: int32")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
        std::vector<string> branchNames;
        TF_RETURN_IF_ERROR(c->GetAttr("branches",&branchNames));
        unsigned int size = 0;
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
                
                size += std::stol(std::string(p1+1,p2));
            }
        }
        //shape_inference::ShapeHandle s = c->MakeShape({c->MakeDim(branchNames.size())});
        shape_inference::ShapeHandle s1 = c->MakeShape({-1,c->MakeDim(size)});
        c->set_output(0, s1);
        
        shape_inference::ShapeHandle s2 = c->MakeShape({-1,1});
        c->set_output(1, s2);
        return Status::OK();
    })
    .Doc(R"doc(A Reader that outputs the lines of a file delimited by '\n'.)doc");

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
                
                virtual void setBranchAddress(TTree* tree) = 0;
                virtual unsigned int fillTensor(typename TTypes<OUT>::Flat& flatTensor, unsigned int index, const OUT& reset) const = 0;
        };
        
        
        template<typename IN, typename OUT=IN>
        class TensorFillerTmpl:
            public TensorFiller<OUT>
        {
            private:
                unsigned int size_;
                std::unique_ptr<TTreeFormula> formula_;
            public:
                typedef TensorFiller<OUT> Base;
            
                TensorFillerTmpl(const string& name, const string& expr, unsigned int size):
                    TensorFiller<OUT>(name,expr),
                    size_(size),
                    formula_(nullptr)
                {
                }
                
                virtual ~TensorFillerTmpl()
                {
                }
                
                virtual void setBranchAddress(TTree* tree)
                {
                    const long id = std::hash<std::thread::id>()(std::this_thread::get_id());
                    formula_.reset(new TTreeFormula((Base::name()+std::to_string(id)).c_str(),Base::expr().c_str(),tree));
                    
                    if(not formula_)
                    {
                        throw std::runtime_error("Cannot parse equation '"+Base::expr()+"'");
                    }
                    formula_->SetQuickLoad(true);
                }
                
                virtual unsigned int fillTensor(typename TTypes<OUT>::Flat& flatTensor, unsigned int index, const OUT& reset) const
                {
                    unsigned int leafSize = formula_->GetNdata(); //needs to be called; otherwise elements >0 are set to 0
                    for (unsigned int i = 0; i < std::min<unsigned int>(leafSize,size_); ++i)
                    {
                        //std::cout<<TensorFiller<OUT>::expr()<<std::endl;
                        double result = formula_->EvalInstance(i);
                        flatTensor(index+i)=Base::resetNanOrInf(result,reset);
                    }
                    for (unsigned int i = std::min(leafSize,size_); i < size_; ++i)
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
        std::vector<std::shared_ptr<TensorFiller<float>>> tensorFillers_;
        size_t currentEntry_;
        
        int naninf_;
        string treename_;
        unsigned int size_;
        int nBatch_;
        unsigned int nEvents_;
        
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
            //gROOT->gErrorIgenoreLevel = 5000;
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
                        tensorFillers_.emplace_back(std::make_shared<TensorFillerTmpl<float>>(
                            "expr_"+std::to_string(iname),
                            name,
                            1
                        ));
                        size_+=1;
                    }
                    else
                    {
                        string type(it+1,name.end());
                        if (type=="UInt_t")
                        {

                            tensorFillers_.emplace_back(std::make_shared<TensorFillerTmpl<unsigned int, float>>(
                                "expr_"+std::to_string(iname),
                                string(name.begin(),it),
                                1
                            ));
                            size_+=1;
                        }
                    }
                }
                else
                {
                    auto p1 = std::find(name.begin(),name.end(),'[');
                    auto p2 = std::find(p1,name.end(),']');
                    std::string branchName(name.begin(),p1);
                    unsigned int size = std::stol(std::string(p1+1,p2));
                    size_+=size;
                    tensorFillers_.emplace_back(
                        std::make_shared<TensorFillerTmpl<float>>(
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
        }
        
        void Compute(OpKernelContext* context)
        {
            mutex_lock localLock(localMutex_);
           
            while (not inputFile_)
            {
                QueueInterface* queue;
                OP_REQUIRES_OK(context,GetResourceFromContext(context, "queue_handle", &queue));
                
                string fileName = GetNextFilename(queue,context);
                if (!context->status().ok())
                {
                    return; //status is bad when queue is closed, so no more reduce_files -> training has finished
                }
                if (fileName.size()==0) throw std::runtime_error("Got empty filename");
                   
                //creating TFile/setting branch adresses is not thread safe
                RootMutex::Lock lock = RootMutex::lock();
                //TODO: use TF logging and set loglevel
                //std::cout<<"opening file "<<fileName<<std::endl;
                TFile* f = TFile::Open(fileName.c_str());
                if (not (f and f->IsOpen()))
                {
                    throw std::runtime_error("Cannot read file '"+fileName+"'");
                }
                //std::cout<<"-> sucessfully open file "<<fileName<<std::endl;
                inputFile_.reset(f);
                
                currentEntry_ = 0;
                tree_ = dynamic_cast<TTree*>(inputFile_->Get(treename_.c_str()));
                if (not tree_)
                {
                    throw std::runtime_error("Cannot get tree '"+treename_+"' from file '"+fileName+"'");
                }
                for (auto& tensorFiller: tensorFillers_)
                {
                    tensorFiller->setBranchAddress(tree_);
                }
                nEvents_ = tree_->GetEntries();
            }
            Tensor* output_tensor = nullptr;
            TensorShape shape;
            unsigned int nBatches = std::min<unsigned int>(nEvents_-currentEntry_,nBatch_);
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
            unsigned int index = 0;
            
            for (unsigned int ibatch=0; ibatch<nBatches;++ibatch)
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
                //inputFile_->Close(); //sometimes this yields a segfault from root
                inputFile_.reset(nullptr);
            }
        }
        
        string GetNextFilename(QueueInterface* queue, OpKernelContext* context) const 
        {
            //mutex_lock localLock(localMutex_); //mutex here makes deadlock for some reason
            //TODO: check core/framework/reader_base.cc for details
            string work;
            Notification n;
            queue->TryDequeue(
                context, [this, context, &n, &work](const QueueInterface::Tuple& tuple) 
                {
                    if (context->status().ok())
                    {
                        if (tuple.size() != 1) 
                        {
                            context->SetStatus(errors::InvalidArgument("Expected single component queue"));
                        } 
                        else if (tuple[0].dtype() != DT_STRING) 
                        {
                            context->SetStatus(errors::InvalidArgument("Expected queue with single string component"));
                        } 
                        else if (tuple[0].NumElements() != 1) 
                        {
                            context->SetStatus(errors::InvalidArgument("Expected to dequeue a one-element string tensor"));
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

//mutex RootReaderOp::localMutex_;


REGISTER_KERNEL_BUILDER(Name("RootReader").Device(DEVICE_CPU),RootReaderOp);

