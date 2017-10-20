
clc;
clear;

load('../data/mnist_uint8.mat');
% test_x (10000 x 784)
% test_y (10000 x 10)
% train_x (60000 x 784)
% train_y (60000 x 10)
test_x=double(test_x)/255;
test_y=double(test_y);
train_x=double(train_x)/255;
train_y=double(train_y);

iter_num =10000;
train_size = length(train_x);
batch_size = 30;

lr= 0.1;
input = 784;

input_size =784;
hidden1_size =100;
hidden2_size =50;
output_size =10;

MLP.layers = {
    struct('func',@Affine, 'type', 'Affine','in',input_size,'out',hidden1_size) 
    struct('func',@Relu, 'type', 'Relu') 
    struct('func',@BatchNorm, 'type', 'BatchNorm','size',hidden1_size) 
    struct('func',@Dropout,'type', 'Dropout', 'dropout_ratio', 0.5) 
    
    struct('func',@Affine,'type', 'Affine', 'in', hidden1_size, 'out', hidden2_size) 
    struct('func',@Relu,'type', 'Relu') 
    
    struct('func',@Affine,'type', 'Affine', 'in', hidden2_size,'out',output_size) 
    struct('func',@SoftmaxWithLoss,'type', 'SoftmaxWithLoss') 
};

network = MultiLayerNet(MLP.layers);

iter_per_epoch = round(train_size/batch_size);



for i = 1: iter_num
    
    batch_mask = randperm(train_size,batch_size);
    x_batch = train_x(batch_mask,:);
    t_batch = train_y(batch_mask,:);
    
    [params_grads,BN_grads] = network.gradient(x_batch,t_batch);
    
    %network.SDG_update(params_grads,BN_grads,lr);
    %network.Momentum_update(params_grads,BN_grads,lr);
    network.AdaGrad_update(params_grads,BN_grads,lr);
    
    loss = network.loss(x_batch,t_batch);
    
    if mod(i,100) == 0
        train_acc = network.accuracy(train_x,train_y)
        test_acc = network.accuracy(test_x,test_y);
        
    end
    
end
