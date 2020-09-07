LOCAL = pwd;
cd ..
cd data
cd extracted
load CHUXING
load QIANRU
load QIANXI
load XINZENG
QIANXI(isnan(QIANXI)) = 0;
external_risk = nan(91,101);
for i=1:91
    external_risk(i,:) = QIANRU(i,:).*(XINZENG(i,:)*permute(QIANXI(i,:,:),[2,3,1]));
end
internal_risk = CHUXING.*XINZENG;
save internal_risk.mat internal_risk
save external_risk.mat external_risk
