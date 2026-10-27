suppressPackageStartupMessages({library(lme4)})
# ====== GLMM 适用性/假设诊断（主分析 Model A，全量剔Ac/Am）======
dat_path <- Sys.getenv("GLME_DATA", "/home/jyy/Documents/Beh_Ana_J/glme/glme_trial_data.csv")
d <- read.csv(dat_path, stringsAsFactors=FALSE)
d$MouseID <- factor(d$MouseID)
d$ProbabilityCondition <- relevel(factor(d$ProbabilityCondition, levels=c("6:4","7:3","8:2","9:1")), ref="7:3")
d$PreviousOutcome <- factor(d$PreviousOutcome, levels=c(0,1))
d$Restriction <- factor(d$Restriction, levels=c("F","W"))
d$Trial_s <- d$Trial / 1000
d$Session_s <- d$SessionNum / 10
d$prefix <- sub("\\d[FW]$", "", d$Treatment)
d <- d[!(d$prefix %in% c("Ac","Am")),]
cat(sprintf("诊断数据: %d trials, %d 鼠\n", nrow(d), nlevels(droplevels(d$MouseID))))

fe <- "Trial_s + Session_s + ProbabilityCondition + PreviousOutcome + Restriction"
fA <- as.formula(paste0("ChoseHighProb ~ ", fe, " + (Trial_s|MouseID)"))
mA <- glmer(fA, data=d, family=binomial(link="logit"),
            control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=1e5)))
m0 <- glmer(as.formula(paste0("ChoseHighProb ~ ", fe, " + (1|MouseID)")), data=d,
            family=binomial(link="logit"),
            control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=1e5)))

cat("\n========== 1. 随机效应结构 (LRT: 随机斜率 vs 仅截距) ==========\n")
lrt <- anova(m0, mA)
print(lrt)
# 随机斜率方差是否>0（边界校正：H0 方差=0 在参数空间边界，p 应 0.5 χ²0 + 0.5 χ²1）
vc <- VarCorr(mA)$MouseID
cat(sprintf("随机截距 SD=%.4f  随机斜率 SD=%.4f  截距-斜率相关=%.3f\n",
    attr(vc,"stddev")[[1]], attr(vc,"stddev")[[2]],
    attr(vc,"correlation")[1,2]))
cat(sprintf("普通 LRT p=%.3g；边界校正后 p≈%.3g（0.5×χ²0+0.5×χ²1）\n",
    lrt$`Pr(>Chisq)`[2], 0.5*lrt$`Pr(>Chisq)`[2]))

cat("\n========== 2. 随机效应正态性 (BLUP) ==========\n")
rr <- ranef(mA)$MouseID
sh1 <- shapiro.test(rr[,1]); sh2 <- shapiro.test(rr[,2])
cat(sprintf("随机截距: Shapiro W=%.3f p=%.3g %s\n", sh1$statistic, sh1$p.value, ifelse(sh1$p.value<0.05,"(偏正态)","(正态)")))
cat(sprintf("随机斜率: Shapiro W=%.3f p=%.3g %s\n", sh2$statistic, sh2$p.value, ifelse(sh2$p.value<0.05,"(偏正态)","(正态)")))

cat("\n========== 3. 共线性 (VIF) ==========\n")
X <- model.matrix(~ Trial_s + Session_s + ProbabilityCondition + PreviousOutcome + Restriction, data=d)
X <- X[, -1]  # 去截距
vif <- sapply(colnames(X), function(v){
  others <- setdiff(colnames(X), v)
  r2 <- summary(lm(X[,v] ~ X[, others]))$r.squared
  1/(1-r2)
})
print(round(vif, 3))
cat(sprintf("最大 VIF=%.2f %s（>5 多重共线，>10 严重）\n", max(vif), ifelse(max(vif)>5,"⚠","✓")))

cat("\n========== 4. 过度离散 (Dispersion) ==========\n")
# 二项 GLMM: 分组聚合算 Pearson χ²/df（按 MouseID×条件×Session bin）
d$grp <- interaction(d$MouseID, d$ProbabilityCondition, d$PreviousOutcome, d$Restriction, drop=TRUE)
pred <- predict(mA, type="response")
agg <- aggregate(cbind(y=ChoseHighProb, phat=pred) ~ grp, data=d, FUN=function(x) mean(x))
agg$n <- as.numeric(table(d$grp))
# 用原始分组（每组 n trial）算
agg2 <- data.frame(grp=names(table(d$grp)), n=as.numeric(table(d$grp)))
agg2$y <- as.numeric(tapply(d$ChoseHighProb, d$grp, mean))
agg2$phat <- as.numeric(tapply(pred, d$grp, mean))
pearson <- sum((agg2$n * (agg2$y - agg2$phat)^2) / (agg2$n * agg2$phat * (1-agg2$phat)))
df_disp <- nrow(agg2) - length(fixef(mA))
cat(sprintf("分组数=%d  Pearson χ²=%.1f  χ²/df=%.3f %s\n",
    nrow(agg2), pearson, pearson/df_disp,
    ifelse(pearson/df_disp > 1.5, "⚠ 过度离散", ifelse(pearson/df_disp < 0.8, "欠离散", "✓ 离散合理"))))

cat("\n========== 5. 拟合优度 (AUC / 预测准确率) ==========\n")
# AUC (rank-based): case=ChoseHighProb=1
pos_rk_sum <- sum(rank(pred)[d$ChoseHighProb==1])
n1 <- as.numeric(sum(d$ChoseHighProb==1)); n0 <- as.numeric(sum(d$ChoseHighProb==0))
auc <- (pos_rk_sum - n1*(n1+1)/2) / (n1*n0)
acc <- mean((pred > 0.5) == d$ChoseHighProb)
null_rate <- max(mean(d$ChoseHighProb), 1-mean(d$ChoseHighProb))
cat(sprintf("AUC=%.4f  准确率=%.4f  空模型率=%.4f %s\n", auc, acc, null_rate,
    ifelse(auc>0.7,"✓ 区分度可", ifelse(auc>0.6,"弱区分","无区分"))))
# McFadden 伪 R²
ll_full <- as.numeric(logLik(mA))
ll_null <- as.numeric(logLik(glm(ChoseHighProb~1, data=d, family=binomial)))
cat(sprintf("McFadden 伪R²=%.4f（>0.2 优, 0.1-0.2 好, <0.1 弱）\n", 1-ll_full/ll_null))

cat("\n========== 6. 残差模式 (binned residual) ==========\n")
res <- residuals(mA, type="pearson")
# 按 pred 分 20 箱，看残差均值
bins <- cut(pred, breaks=quantile(pred, probs=seq(0,1,length=21), na.rm=TRUE), include.lowest=TRUE)
br <- aggregate(res ~ bins, FUN=function(x) c(mean=mean(x), n=length(x), se=2/sqrt(length(x))))
br <- do.call(data.frame, br)
# 严格阈值 ±2/sqrt(n)（大样本下过严）
over <- sum(abs(br$res.mean) > br$res.se)
# 幅度阈值：残差均值绝对值 >0.1 才算实质偏差
mag_over <- sum(abs(br$res.mean) > 0.1)
cat(sprintf("20箱残差均值范围 [%.3f, %.3f]\n", min(br$res.mean), max(br$res.mean)))
cat(sprintf("严格 ±2/sqrt(n): %d/20 超出（大样本阈值=%.3f 过严）\n", over, br$res.se[1]))
cat(sprintf("幅度 >0.1: %d/20 超出 %s\n", mag_over,
    ifelse(mag_over<=4,"✓ 实质无系统偏差",ifelse(mag_over<=8,"可接受","⚠ 系统偏差"))))

cat("\n========== 7. 样本量充分性 ==========\n")
per_mouse <- table(d$MouseID)
cat(sprintf("每鼠 trial: 中位=%.0f  min=%.0f  max=%.0f\n",
    median(per_mouse), min(per_mouse), max(per_mouse)))
cat(sprintf("参数数(固定)=%d  总 events(=1)=%d  events/参数=%.1f（应>10/参数, 理想>30）\n",
    length(fixef(mA)), sum(d$ChoseHighProb==1), sum(d$ChoseHighProb==1)/length(fixef(mA))))

cat("\n========== 8. ICC (组内相关) ==========\n")
# ICC = 随机截距方差 / (随机截距方差 + π²/3) [logit 残差方差=π²/3]
var_int <- attr(vc,"stddev")[[1]]^2
icc <- var_int / (var_int + pi^2/3)
cat(sprintf("ICC=%.4f（同鼠内 trial 相关性，0.05-0.15 中等, >0.2 强聚集）→ %s\n",
    icc, ifelse(icc>0.1,"需随机效应",ifelse(icc>0.05,"随机效应有益","聚集弱"))))

cat("\n========== 9. 固定效应 Wald vs LRT (稳健性) ==========\n")
# 比较关键项：单项 drop 后 LRT p vs Wald p
# fe 里 term 名是简写（如 PreviousOutcome），系数行名带水平（PreviousOutcome1）
coefnames <- rownames(summary(mA)$coefficients)
drop_one <- function(term){
  fe2 <- paste(setdiff(strsplit(fe," \\+ ")[[1]], term), collapse=" + ")
  m2 <- glmer(as.formula(paste0("ChoseHighProb ~ ", fe2, " + (Trial_s|MouseID)")),
              data=d, family=binomial(link="logit"),
              control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=1e5)))
  a <- anova(m2, mA)
  # 找系数行名：term 本身或以 term 开头（因子水平）
  rn <- grep(paste0("^", term), coefnames, value=TRUE)
  if(length(rn)==0) rn <- term
  wald_p <- min(summary(mA)$coefficients[rn, 4], na.rm=TRUE)
  list(wald=wald_p, lrt=a$`Pr(>Chisq)`[2])
}
for (t in c("Trial_s","Session_s","PreviousOutcome","Restriction")){
  r <- drop_one(t)
  cat(sprintf("%-20s Wald p=%.3g  LRT p=%.3g  %s\n", t, r$wald, r$lrt,
    ifelse(sign(r$wald-0.05)==sign(r$lrt-0.05),"一致","⚠ 不一致")))
}

cat("\n========== 诊断完成 ==========\n")
