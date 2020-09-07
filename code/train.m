cd ..
cd data
cd extracted
load CHUXING
load external_risk
load XINZENG
load QUEZHEN

time = repmat([1:91]', [1,101]);

internal_risk = CHUXING.*XINZENG;

ex_risk_mean = nan(91,101);
in_risk_mean = nan(91,101);
for i=7:91
    ex_risk_mean(i,:) = mean(external_risk(i-6:i-3,:));
    in_risk_mean(i,:) = mean(internal_risk(i-6:i-3,:));
end

except_WH = [1:56,58:101];
R = except_WH(randperm(length(except_WH)));
batch_size = floor(length(R)/10);
mu_X = cell(10,1);
mu_Y = cell(10,1);
std_Y = cell(10,1);
std_X = cell(10,1);
nets = cell(10,1);

test_set = nan(91,101);
for k=1:10
    test_cities = R((k-1)*batch_size+1:k*batch_size);
    train_cities = setdiff(except_WH, test_cities);
    
    X = [...
        reshape(ex_risk_mean(1:end-1, train_cities),[],1),...
        reshape(in_risk_mean(1:end-1, train_cities),[],1),...
        reshape(time(1:end-1, train_cities),[],1)...
    ];
    Y = reshape(XINZENG(2:end, train_cities),[],1);
    X(isnan(Y),:) = [];
    Y(isnan(Y)) = [];
    for t=1:2
        Y(isnan(X(:,t))) = [];
        X(isnan(X(:,t)),:) = [];
    end
    mu_X{k} = mean(X, 1);
    std_X{k} = std(X);
    mu_Y{k} = mean(Y);
    std_Y{k} = std(Y);
    norm_X = (X-mu_X{k})./std_X{k};
    norm_Y = (Y-mu_Y{k})./std_Y{k};
    
    n = cell(5,1);
    for t=1:5
        net = feedforwardnet([5,3]);
        net.trainParam.showWindow = 0;
        [net, tr] = train(net, norm_X', norm_Y');
        n{t} = net;
    end
    nets{k} = n;
    
    for p=test_cities
        X = [...
            reshape(ex_risk_mean(1:end-1,p),[],1),...
            reshape(in_risk_mean(1:end-1,p),[],1),...
            reshape(time(1:end-1,p),[],1)...
        ];
        norm_X = (X-mu_X{k})./std_X{k};
        temp_pred = [];
        for t=1:5
            temp_pred(:,t) = n{t}(norm_X');
        end
        test_set(2:end,p) = mean(temp_pred,2).*std_Y{k}+mu_Y{k};
    end
end
test_set(23,except_WH) = QUEZHEN(23,except_WH);
test_set(24:30,except_WH) = XINZENG(24:30,except_WH);
test_set(isnan(test_set)) = 0;
test_set(test_set<0) = 0;
pred_all_cities = cumsum(test_set);
plot(nansum(pred_all_cities,2));
hold on
plot(nansum(QUEZHEN(:,except_WH),2));
cd ..
cd ..
save nets.mat nets mu_X mu_Y std_X std_Y
