#include "neuronnet.h"


namespace Net
{

    NeuronNet::NeuronNet() {}
    NeuronNet::~NeuronNet() {}

    /// @brief main neuron net processes
    void NeuronNet::main(MLayer &request ,const std::string &model)
    {
        this->dRead("nWeights");

        this->dGet(request , model);

        this->nQuery(request);

        this->nFinalization();
    }

    /// @brief main neuron net processes , query with train
    void NeuronNet::main(DataSet &data , const std::string &model)
    {
        this->dRead("nWeights");
        this->dGet(data.first ,model);

        this->nQuery(data.first);

        for(uint_fast16_t i = 0; i < pStandart; i++)
        {
            this->nTrain(data.second);
        }

        this->dUpdate(model);
        this->dWrite("nWeights");

        this->nQuery(data.first);
        this->nFinalization();
    }


    /// init of neuron net
    void NeuronNet::nInit(Weight &w1, Weight &w2)
    {
        /// @brief init weights of neuron net
        this->inpToHidW = Weight(w1);
        this->hidToOutW = Weight(w2);
    }

    void NeuronNet::nReInit()
    {
        for(size_t i =0; i< inpToHidW.size1();i++){
            for(size_t j =0; j< inpToHidW.size2();j++){
                inpToHidW(i,j) = 0.5;
            }
        }

        for(size_t i =0; i< hidToOutW.size1();i++){
            for(size_t j =0; j< hidToOutW.size2();j++){
                hidToOutW(i,j) =  0.5;
            }
        }
    }

    void NeuronNet::nRandInit(Weight &w1, Weight &w2)
    {
        std::random_device rd;
        std::mt19937 mersenne(rd());

         for(size_t i =0; i< w1.size1();i++){
             for(size_t j =0; j< w1.size2();j++){
                 w1(i,j) = mersenne()/double(RAND_MAX);
             }
         }

         for(size_t i =0; i< w2.size1();i++){
             for(size_t j =0; j< w2.size2();j++){
                 w2(i,j) =  mersenne()/double(RAND_MAX);
             }
         }

         this->nInit(w1,w2);

    }


    void NeuronNet::nQuery(mat::matrix<double> &param)
    {



        /// @brief
        /// init input matrix layer , calculation hidden matrix
        /// layer and call activate of hidden neurons
        this->inpMLayer = MLayer(param);

        this->hidMLayer = prod(inpToHidW , inpMLayer);
            for(uint_fast16_t i =0; i< hidMLayer.size1();i++)
            {
                for(uint_fast16_t j =0; j< hidMLayer.size2();j++)
                {
                    hidMLayer(i,j) = tanhf64(hidMLayer(i,j));
                }
            }


        /// @brief calculation output matrix layer and call activate of output neurons
        this->outMLayer = prod(hidToOutW, hidMLayer);
            for(uint_fast16_t i =0; i< outMLayer.size1();i++)
            {
                for(uint_fast16_t j =0; j< outMLayer.size2();j++)
                {
                    outMLayer(i,j) = tanhf64(outMLayer(i,j));
                }
            }


    }


    void NeuronNet::nTrain(MLayer& target)
    {

        /// @brief activate hidden layer
        this->hidMLayer = prod(inpToHidW , inpMLayer);
        for(uint_fast16_t i =0; i< hidMLayer.size1();i++)
        {
            for(uint_fast16_t j =0; j< hidMLayer.size2();j++)
            {
                hidMLayer(i,j) = tanhf64(hidMLayer(i,j));
            }
        }



        /// @brief activate output layer
        this->outMLayer = prod(hidToOutW, hidMLayer);
        for(uint_fast16_t i =0; i< outMLayer.size1();i++)
        {
            for(uint_fast16_t j =0; j< outMLayer.size2();j++)
            {
               outMLayer(i,j) = tanhf64(outMLayer(i,j));
            }
        }



        /// @brief calculate error on hidden and output layer
        MLayer outErrors(target.size1() , target.size2());
        for(uint_fast16_t i =0; i< outMLayer.size1();i++)
        {
            for(uint_fast16_t j =0; j< outMLayer.size2();j++)
            {
               outErrors(i,j) = target(i,j) - outMLayer(i,j);
            }
        }

        MLayer hidErrors =  mat::prod(mat::trans(this->hidToOutW),outErrors);


        /// @brief update coefficients between hidden and output layers
        for(uint_fast16_t i = 0; i < outMLayer.size1(); i++)
        {
            for(uint_fast16_t j = 0; j < outMLayer.size2(); j++)
            {
                  this->outMLayer(i,j) = 2*outErrors(i,j)*(1 - pow(tanhf64(outMLayer(i,j)),2));
            }
        }
        this->hidToOutW += mat::prod(this->outMLayer , mat::trans(hidMLayer));

        /// @brief update coefficients between input and hidden layers
        for(uint_fast16_t i = 0; i < hidMLayer.size1(); i++)
        {
            for(uint_fast16_t j = 0; j < hidMLayer.size2(); j++)
            {
                  this->hidMLayer(i,j) = 2*hidErrors(i,j)*(1 - pow(tanhf64(hidMLayer(i,j)),2));
            }
        }
        this->inpToHidW += mat::prod(this->hidMLayer , mat::trans(inpMLayer));

    }

    void NeuronNet::dUpdate(const std::string &bType)
    {
        pt::ptree target  = _root.get_child(bType);

        pt::ptree::assoc_iterator iterator1 = target.find("inputWeights");
        pt::ptree::assoc_iterator iterator2 = target.find("hiddenWeights");

        iterator1->second.erase(iterator1->second.begin(), iterator1->second.end());
        iterator2->second.erase(iterator2->second.begin(), iterator2->second.end());

        for (uint_fast16_t i = 0; i < this->inpToHidW.size1(); i++)
        {
            pt::ptree row;

            for(uint_fast16_t j =0; j< this->inpToHidW.size2(); j++)
            {
                 pt::ptree cell;
                 cell.put_value(this->inpToHidW(i,j));
                 row.push_back(std::make_pair("", cell));
            }
            iterator1->second.push_back(std::make_pair("", row));
        }





        for (uint_fast16_t i = 0; i < this->hidToOutW.size1(); i++)
        {
            pt::ptree row;

            for(uint_fast16_t j =0; j< this->hidToOutW.size2(); j++)
            {
                 pt::ptree cell;
                 cell.put_value(this->hidToOutW(i,j));
                 row.push_back(std::make_pair("", cell));
            }
            iterator2->second.push_back(std::make_pair("", row));
        }

         this->dWrite("nWeights");
    }

    void NeuronNet::dGet(MLayer& request, const std::string &bType)
    {


        Weight w1(request.size1(),request.size1());
        Weight w2(request.size1(),request.size1());


            /// @brief get weights from json file
            pt::ptree::assoc_iterator target  = _root.find(bType);

            /// @brief if weights not found -> start generation
            /// of new neuron net for this model
            if(target->first.empty())
            {
                this->dCreate(request , bType);
            }
            /// @def get weights from json file
            _children = _root.get_child(bType);

            /// fill weights from getted data from json file
            uint_fast16_t x = 0;
            for (pt::ptree::value_type &row : _children.get_child("inputWeights"))
            {
                uint_fast16_t y = 0;
                for (pt::ptree::value_type &cell : row.second)
                {
                    if(x >= request.size1() or y == request.size1()) { break ; }
                    w1(x,y) = cell.second.get_value<double>();
                    y++;
                }
                x++;
            }

            x = 0;
            for (pt::ptree::value_type &row : _children.get_child("hiddenWeights"))
            {
                uint_fast16_t y = 0;
                for (pt::ptree::value_type &cell : row.second)
                {
                    if(x >= request.size1() or y == request.size1()) { break ; }
                    w2(x,y) = cell.second.get_value<double>();
                    y++;
                }
                x++;
            }

        /// @brief if request size > w1&w2 size`s -> extend w1&w2 size
        if(request.size1() != w1.size1() && request.size1() != w1.size2())
        {
            this->meiosis(w1,w2,request);
            this->nInit(w1,w2);
            this->dUpdate(bType);

            return void();
        }

        this->nInit(w1,w2);

    }

    /// generator neuron net`s
    void NeuronNet::dCreate(MLayer &request, const std::string &bType)
    {
        /// @def
        std::random_device rd;
        std::mt19937 mersenne(rd());
        pt::ptree Weights;


        pt::ptree weight;
        for (uint_fast16_t x = 0; x < request.size1(); x++)
        {
            pt::ptree row;
            for(uint_fast16_t y =0; y< request.size2(); y++)
            {
                 pt::ptree cell;
                 cell.put_value(mersenne()/double(RAND_MAX));
                 for(uint_fast16_t z  = 0; z < request.size1(); z++)
                 {
                     row.push_back(std::make_pair("", cell));
                 }
            }
            weight.push_back(std::make_pair("", row));
        }
        Weights.push_back(std::make_pair("inputWeights", weight));


        weight.clear();
        for (uint_fast16_t x = 0; x < request.size1(); x++)
        {
            pt::ptree row;
            for(uint_fast16_t y =0; y< request.size2(); y++)
            {
                 pt::ptree cell;
                 cell.put_value(mersenne()/double(RAND_MAX));
                 for(uint_fast16_t z  = 0; z < request.size1(); z++)
                 {
                     row.push_back(std::make_pair("", cell));
                 }
            }
            weight.push_back(std::make_pair("", row));
        }
        Weights.push_back(std::make_pair("hiddenWeights", weight));

        _root.add_child(bType, Weights);
        this->dWrite("nWeights");

    }



    /// @brief this method droped weights and setup default weights ,
    ///  after that started training neuron net on new. Also we can
    ///  setup precision of train neuron net.
    void NeuronNet::nReTrain(MLayer &target , int precision)
    {
        this->nReInit();
        for(uint_fast8_t x = 0; x < precision; x++)
        {
              this->nTrain(target);
        }

    }

    /// @todo need to rewrite this method
    void NeuronNet::nFinalization()
    {
        std::random_device rd;
        std::mt19937 mersenne(rd());
        std::vector<uint_fast16_t> keys;


        for(uint_fast16_t x = 0; x < outMLayer.size1(); x++)
        {
            for(uint_fast16_t y = 0; y < outMLayer.size2(); y++)
            {
                  outMLayer(x,y) =  round(outMLayer(x,y)*mersenne()/(1.7*double(RAND_MAX)));
            }
        }


        for(uint_fast16_t x = 0; x < outMLayer.size1(); x++)
        {
            for(uint_fast16_t y = 0; y < outMLayer.size2(); y++)
            {
                  if(outMLayer(x,y) >= 1)
                  {
                      keys.push_back(x);
                  }
            }
        }

        this->aKey = static_cast<int>(keys[mersenne()%keys.size()]);


    }



}
