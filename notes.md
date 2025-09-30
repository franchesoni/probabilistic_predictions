
# mcdpout paper
I should cite neal 1995 and MacKay 1992 for bayesian nn
Apparently bayesian networks have challenging inference and additional computational
costs.
It seems that Monte Carlo dropout is not achieved by simply ranking.



# A survey of uncertainty in deep neural networks (2023)
The most important approaches for modeling this separation are Bayesian inference
(Blundell et al. 2015; Gal and Ghahramani 2016; Mobiny et al. 2021; Amini et al. 2018;
Krueger et al. 2017), ensemble approaches (Lakshminarayanan et al. 2017; ValdenegroToro 2019; Wen et al. 2019), test-time augmentation approaches (Shorten and Khoshgoftaar 2019; Wen et  al. 2021a), or single deterministic networks containing explicit components to represent the model and the data uncertainty (Malinin and Gales 2018; Sensoy et al. 2018; Malinin and Gales 2019; Raghu et al. 2019).

s (Lakshminarayanan et  al. 2017), for ensembles

In general, the methods for estimating the uncertainty can be split into four diferent
types based on the number (single or multiple) and the nature (deterministic or stochastic)
of the used DNNs.
single deterministic, bayesian, ensembles, test time aug

Gradient metrics. Additional network for uncertainty. Distance to training data.
Prior networks. evidential neural networks. gradient penalties.

oberdiek 2018 extra
lee 2020 extra
raghu 2019 extra
ramalho 2020
amersfoort 2020
malinin 2018 dirichlet
sensoy 2018 binary dirichlet
amersfoort 2020
Możejko et al. 2018; binary
Nandy et al. 2020 dirichlet
Oala et al. 2020)

But in general, this is still more efcient
than the number of predictions needed for ensembles based methods (Sect. 3.3), Bayesian
methods (Sect. 3.2), and test-time data augmentation methods (Sect. 3.4).

A drawback of
single deterministic neural network approaches is the fact that they rely on a single opinion
and can therefore become very sensitive to the underlying network architecture, training
procedure, and training data.

(Denker et al. 1987; Tishby et al. 1989; Buntine and
Weigend 1991) for bayesian

# A review of uncertainty quantification in deep learning: Techniques, applications and challenges 
doesn't mention crps

# energy based prob regression
. The
discretization of the target space Y however complicates exploiting its inherent
neighborhood structure, an issue that has been addressed by exploring ordinal
regression methods for 1D problems [4, 11]. 


A General Framework
for Uncertainty Estimation in Deep Learning

they compute droput rates after training via neg log lik

# A Large-Scale Study of Probabilistic Calibration in Neural Network Regression

- (Tagasovska and Lopez-Paz, 2019; Chung et al., 2021; Feldman et al., 2021) are also references for implicit quantile network.
    - The model is trained by minimizing the quantile score at multiple levels, which is asymptotically equivalent to minimizing the CRPS (Bracher et al., 2021).

- the following is so strange: is like calibration given performance, not viceversa.
For regularization methods,
an important hyperparameter is the regularization factor λ.
As previously observed in classification (Karandikar et al.,
2021), we found that higher values of λ tend to improve
calibration but worsen NLL, CRPS, and STD. Karandikar
et al. (2021) proposed to limit the loss in accuracy by a
maximum of 1%. We adopt a similar strategy by selecting
λ which minimizes PCE with a maximum increase in CRPS
of 10% in the validation set. F
- Note that we only consider MIX models since we cannot compute the NLL for SQR. (this is wrong!)

I read some nearest neighbors to our paper. In particular two surveys [1,2] and two papers regarding regression [3,4].
What did I learn? The first thing I learned is that these two recent
surveys, do not really give a lot of attention to predicting Probability distributions and to predict them directly. One of them considers what they
called single deterministic predictors but in fact they do not consider many of the losses we propose. The most alarming thing of this survey is that they do not introduce the CRPS as a metric
and they don't really focus on regression. 
In contrast, the last 2 papers do focus on regression.
The first one provides an excellent related work section that could eclipse our own paper.It also introduces an energy-based network that is analogous to the implicit quantile network,
but instead of predicting the quantile, it predicts the density.
In other words, the implicit quantile network is to quantile regression,
what this proposed energy-based network is to the histogram estimation.
The last paper talks mostly about calibration for regression and it does
experiments on a very large number of datasets and it does introduce the CRPS. It
does use a negative log likelihood and it does use two of the models that we
introduced. In particular they use the mixture density network optimized
with the negative log likelihood and also the method with the CRPS which we
didn't propose unless they also propose quantile regression scheme and they
showed that minimizing the CRPS is equivalent asymptotically to minimizing
the pinball loss.
In fact, an our variation that we could introduce is to do histogram estimation,
minimizing the CRPS.




we have created a taxonomy:
- crps vs logscore
- parametric vs PL
- height vs quantile
- implicit vs explicit


what recommendations can we make?
- naturally, we would like to see that minimizing score S results in the best test performance as measured by S. (this has a name, couldn't find it).
    - If it happens, we look for the name of this property and that's about it. We comment on the advantages of CRPS / logscore as metrics (low prob behavior) and refer to the specific application.
    - If it doesn't, we see if CRPS is always better.
        - if it is, we cite the cites in energy based methods above about neighborhood info exploitation
        - if logscore is always better, we plot the gradients of the crps and logscore (CE) and think about a next step.
        - if they are inverted, e.g. logscore helps crps and viceversa, look at the bishop toy experiment
- for parametric vs pl, we need to see which one wins for each dataset
    - if there is no consistent winner, we must order the datasets by size
        - if there's no relation to data size, we can try to assess multimodality (e.g. using two gaussians vs one gaussian)
            - if we don't find anything, ask JM
- for height vs. quantile, the main difference goes on how many quantiles does one use, the need for defining the upper and lower bounds, and the amount of data. Conclusions must hold across both objectives (CE / histcrps / energy vs pinball / crpsqr / iqn) should work well.
- for implicit vs explicit one should consider that implicit has potentially more representation power, as one doesn't constrain to only some points, but looks at all points of the CDF. However, it might be harder to train.


    



other notes:
- compute the skill for each dataset

results
mcd_Abalone
Final results:
Logscore:
-0.785552442073822, CRPS:
0.04069489613175392, ECE: 
0.0961744412779808

mcd_Concrete Compressive Strength
Final results:
Logscore: 
-0.7493574619293213, CRPS: 
0.06335991621017456, ECE: 
0.03806642070412636

mcd_Energy Efficiency
Final results:
Logscore: 
-1.128684401512146, CRPS: 
0.04289770871400833, ECE: 
0.052702706307172775

mcd_Liver Disorders
Final results:
Logscore: 
1.0825471878051758, CRPS: 
0.09403250366449356, ECE: 
0.18057124316692352

mcd_Parkinsons Telemonitoring
Final results:
Logscore: 
-0.013763457536697388, CRPS: 
0.12243806570768356, ECE: 
0.03999200835824013