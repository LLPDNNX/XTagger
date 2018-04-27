#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/op.h"

#include <regex>

using namespace tensorflow;

//TODO: somehow need to parse pt for weight evaluation!!!

REGISTER_OP("ClassificationWeights")
    .Input("labels: float32")
    .Input("input: float32")
    .Output("out: float32")
    .Attr("rootfile: string")
    .Attr("histnames: list(string)")
    .Attr("varindex: list(int)")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
        tensorflow::shape_inference::ShapeHandle label_shape =  c->input(0);
        int batch_dim = c->Value(c->Dim(label_shape,0));
        shape_inference::ShapeHandle s = c->MakeShape({batch_dim});
        c->set_output(0,s);
        return Status::OK();
    })
    .Doc(R"doc(A Reader that outputs the lines of a file delimited by '\n'.)doc");

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "TH2F.h"
#include "TFile.h"

#include "RootMutex.h"


class ClassificationWeightsOp:
    public OpKernel
{

    private:
        std::string filePath;
        std::vector<std::string> histNames;
        bool transpose_;
        std::vector<std::shared_ptr<TH2F>> hists;
        std::vector<int> varIndex;
        
    public:
        explicit ClassificationWeightsOp(OpKernelConstruction* context): 
            OpKernel(context)
        {
            OP_REQUIRES_OK(
                context,
                context->GetAttr("rootfile",&filePath)
            );
            OP_REQUIRES_OK(
                context,
                context->GetAttr("histnames",&histNames)
            );
            OP_REQUIRES_OK(
                context,
                context->GetAttr("varindex",&varIndex)
            );
            RootMutex::Lock lock = RootMutex::lock();
            TFile rootFile(filePath.c_str());
            if (not rootFile.IsOpen ())
            {
                throw std::runtime_error("Root file '"+filePath+"' cannot be opened");
            }
            for (auto histName: histNames)
            {
                TH2F* hist = dynamic_cast<TH2F*>(rootFile.Get(histName.c_str()));
                if (not hist)
                {
                    throw std::runtime_error("Cannot find hist '"+histName+"' in file '"+filePath+"'");
                }
                hist = (TH2F*)hist->Clone();
                hist->SetDirectory(0);
                hists.push_back(std::shared_ptr<TH2F>(hist));
            }
        }
        
        
        virtual ~ClassificationWeightsOp()
        { 
            RootMutex::Lock lock = RootMutex::lock();
            hists.clear();
        }
        
        float computeWeight(int classIndex, float value1, float value2)
        {
            TH2* hist = hists[classIndex].get();
            int bin = hist->FindBin(value1,value2);
            return hist->GetBinContent(bin);
        }

        void Compute(OpKernelContext* context)
        {
            const Tensor& label_tensor = context->input(0);
            auto label = label_tensor.flat<float>();
            long num_batches = label_tensor.dim_size(0);
            long label_length = label_tensor.dim_size(1);
            if (label_length!=hists.size())
            {
                throw std::runtime_error("Labels ("+std::to_string(hists.size())+") need to be of same size as tensor ("+std::to_string(label_length)+")");
            }

            const Tensor& value_tensor = context->input(1);
            auto value = value_tensor.flat<float>();
            long value_size = value_tensor.dim_size(1);

            Tensor* output_tensor = nullptr;
            TensorShape shape;
            shape.AddDim(num_batches);
            OP_REQUIRES_OK(context, context->allocate_output("out", shape,&output_tensor));
            auto output = output_tensor->flat<float>();

            for (unsigned int ibatch = 0; ibatch < num_batches; ++ibatch)
            {
                int class_index = -1;
                for (unsigned int i = 0; i < label_length; ++i)
                { 
                    if (label(ibatch*label_length+i)>0.5)
                    {
                        class_index = i;
                        break;
                    }
                }
                if (class_index<0) throw std::runtime_error("labels tensor needs to be one-hot encoded");
                float varValue1 = value(ibatch*value_size+varIndex[0]);
                float varValue2 = value(ibatch*value_size+varIndex[1]);
                output(ibatch) = computeWeight(class_index,varValue1,varValue2);
            }
        }  
};


REGISTER_KERNEL_BUILDER(Name("ClassificationWeights").Device(DEVICE_CPU),ClassificationWeightsOp);

