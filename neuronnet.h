#pragma once

#include "neuron.h"



namespace Net
{


        ///<summary> define help types
        ///     dataset -> first = request , second = target to train
        /// </summary>         
        typedef std::pair<MLayer , MLayer> DataSet;



        /// @brief
        ///
        /// @file nWeights.json
        ///
        /// n{name} <=> net {method name}
        /// d{name} <=> data {method name}
        ///
        /// @var bType <=> behavior type(name of behavior type)
        /// @class main neuron net



        class NeuronNet : public DataManager , public Cell , public Neuron
        {
            public:
                explicit NeuronNet();
                ~NeuronNet();

                void main(MLayer& request,const std::string &model);
                void main(DataSet& data, const std::string &model);

                void nInit(Weight& w1 , Weight& w2);
                void nReInit();
                void nRandInit(Weight& w1 , Weight& w2);

                void dUpdate(const std::string &bType);
                void dGet(MLayer &request , const std::string &bType);
                void dCreate(MLayer &request , const std::string &bType);

                void nQuery(mat::matrix<double>& param);
                void nTrain(MLayer& target);
                void nReTrain(MLayer& target, int precision = pNormal);

                void nFinalization();

            int    aKey; // activation key

            MLayer inpMLayer;
            MLayer hidMLayer;
            MLayer outMLayer;

            Weight inpToHidW;
            Weight hidToOutW;


        };



}
