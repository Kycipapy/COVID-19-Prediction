cd ..
load net
cd data
cd extracted
load CHUXING
load QIANRU
CHUXING = CHUXING*2;
QIANRU = QIANRU*2;
load QIANXI
QIANXI(isnan(QIANXI)) = 0;
load external_risk
external_risk = external_risk*2;

time = repmat([1:91]',[1,101]);
except_WH = [1:56,58:101];

internal_risk = nan(91,101);

internal_risk(1:30,:) = CHUXING(1:30,:).*XINZENG(1:30,:);
external_risk(31:end,:) = nan;

XINZENG(31:end,except_WH) = nan;
XINZENG(23,:) = QUEZHEN(23,:);

for i=31:91
    ex_risk_mean = mean(external_risk(i-7:i-4,:));
    in_risk_mean = mean(internal_risk(i-7:i-4,:));
    time = (i-1)*ones(1,101);
    X = [ex_risk_mean;in_risk_mean;time]';
    for k=1:10
        norm_X = (X-mu_X{k})./std_X{k};
        temp_pred = nan(101,5);
        for t=1:5
            temp_pred(:,t) = nets{k}{t}(norm_X');
        end
        XINZENG(i,except_WH) = median(temp_pred(except_WH,:),2)'.*std_Y{k}+mu_Y{k};
        external_risk(i,:) = QIANRU(i,:).*(XINZENG(i,:)*permute(QIANXI(i,:,:),[2,3,1]));
        internal_risk(i,:) = CHUXING(i,:).*XINZENG(i,:);
    end
end
XINZENG(isnan(XINZENG)) = 0;
XINZENG(XINZENG<0) = 0;
QUEZHEN = cumsum(XINZENG);
