#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/op.h"

#include <regex>

using namespace tensorflow;


REGISTER_OP("FakeBackground")
    .Input("batch: float32")
    .Input("is_signal: bool")
    .Output("fake_batch: float32")
    .Attr("feature_index: int")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
        c->set_output(0,c->input(0));
        return Status::OK();
    })
    .Doc(R"doc(A Reader that outputs the lines of a file delimited by '\n'.)doc");

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "TH1F.h"
#include "TFile.h"

#include <random>

namespace 
{
    static thread_local std::unique_ptr<std::mt19937> generator_;
}


class FakeBackgroundOp:
    public OpKernel
{

    private:
        int feature_index_;

    public:
        explicit FakeBackgroundOp(OpKernelConstruction* context): 
            OpKernel(context)
        {
            OP_REQUIRES_OK(
                context,
                context->GetAttr("feature_index",&feature_index_)
            );
        }
        
        virtual ~FakeBackgroundOp()
        { 
        }
        
        void Compute(OpKernelContext* context)
        {
            if (not generator_)
            {
                generator_.reset(new std::mt19937(123456)); 
            }
            const Tensor& batch_tensor = context->input(0);
            auto batch = batch_tensor.flat<float>();
            
            const Tensor& is_signal_tensor = context->input(1);
            auto is_signal = is_signal_tensor.flat<bool>();
            
            size_t n_batches = batch_tensor.dim_size(0);
            size_t batch_length = batch_tensor.NumElements()/n_batches;
            if (n_batches!=is_signal_tensor.dim_size(0))
            {
                throw std::runtime_error("Batch ("+
                    std::to_string(n_batches)+
                    ") and label tensor ("+
                    std::to_string(is_signal_tensor.dim_size(0))+
                    ") need to have the same first dimension");
            }
            
            Tensor* fake_batch_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output("fake_batch", batch_tensor.shape(),&fake_batch_tensor));
            auto fake_batch = fake_batch_tensor->flat<float>();
            
            std::vector<float> signal_dist;
            for (size_t ibatch = 0; ibatch < n_batches; ++ibatch)
            {
                if (is_signal(ibatch))
                {
                    signal_dist.push_back(batch(ibatch*batch_length+feature_index_));
                }
            }
            //std::sort(signal_dist.begin(),signal_dist.end());
            
            if (signal_dist.size()==0)
            {
                //just random between -3 and 8
                std::uniform_real_distribution<float> dist(-3,8);

                std::clog << "Warning! No signal in the batch \n" << std::endl;
                for (size_t ibatch = 0; ibatch < n_batches; ++ibatch)
                {
                    for (size_t ielem = 0; ielem < batch_length; ++ielem)
                    {
                        if (ielem == feature_index_)
                        {
                            fake_batch(ibatch*batch_length+ielem) = dist(*generator_);
                        }
                        else
                        {
                            fake_batch(ibatch*batch_length+ielem) = batch(ibatch*batch_length+ielem);
                        }
                    }
                }
            }
            else if (signal_dist.size()==1)
            {
                //set all background to same signal value
                //std::clog << "Warning! Signal distribution has a single value in the batch \n" << std::endl;
                for (size_t ibatch = 0; ibatch < n_batches; ++ibatch)
                {
                    for (size_t ielem = 0; ielem < batch_length; ++ielem)
                    {
                        if (!is_signal(ibatch) and ielem == feature_index_)
                        {
                            fake_batch(ibatch*batch_length+ielem) = signal_dist[0];
                        }
                        else
                        {
                            fake_batch(ibatch*batch_length+ielem) = batch(ibatch*batch_length+ielem);
                        }
                    }
                }
            }
            else //size>1
            {
                // morph between values
                // for displacement use this:
                // std::uniform_real_distribution<float> dist(0,signal_dist.size()-1);
                std::uniform_int_distribution<> dist(0, signal_dist.size()-1);
                for (size_t ibatch = 0; ibatch < n_batches; ++ibatch)
                {
                    for (size_t ielem = 0; ielem < batch_length; ++ielem)
                    {
                        if (!is_signal(ibatch) and ielem == feature_index_)
                        {
                            const float v = dist(*generator_);

                            // for displacement use this instead:
                            //const int l = int(std::floor(v));
                            //const int u = int(std::ceil(v));
                            //const float w = v-l;
                            //fake_batch(ibatch*batch_length+ielem) = signal_dist[l]*w+signal_dist[u]*(1.-w);
                            fake_batch(ibatch*batch_length+ielem) = signal_dist[v];
                        }
                        else
                        {
                            fake_batch(ibatch*batch_length+ielem) = batch(ibatch*batch_length+ielem);
                        }
                    }
                }
            }
        }
};

REGISTER_KERNEL_BUILDER(Name("FakeBackground").Device(DEVICE_CPU),FakeBackgroundOp);

