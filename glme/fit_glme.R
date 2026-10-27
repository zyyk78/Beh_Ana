suppressPackageStartupMessages({library(lme4); library(MASS)})
# 数据路径可被环境变量 GLME_DATA 覆盖(筛选子集时用), 默认全量
dat_path <- Sys.getenv("GLME_DATA", "/home/jyy/Documents/Beh_Ana_J/glme/glme_trial_data.csv")
cat(sprintf("拟合数据: %s\n", dat_path))
d <- read.csv(dat_path, stringsAsFactors=FALSE)
d$MouseID <- factor(d$MouseID)
d$ProbabilityCondition <- relevel(factor(d$ProbabilityCondition,
    levels=c("6:4","7:3","8:2","9:1")), ref="7:3")
d$PreviousOutcome <- factor(d$PreviousOutcome, levels=c(0,1))
d$Restriction <- factor(d$Restriction, levels=c("F","W"))   # F=限食(ref) W=限水
d$Age <- as.numeric(d$Age)
d$Sex  <- factor(d$Sex, levels=c(0,1))                       # 0=雌(ref) 1=雄
d$Strain <- factor(d$Strain, levels=c("WT","Thy1"))          # WT(ref) Thy1

# Trial 全局累计 -> 每1000 trial 一个单位(系数可读)
d$Trial_s <- d$Trial / 1000
# Session(第几次训练, within-mouse训练经验) -> 每10 session 一个单位
d$Session_s <- d$SessionNum / 10
d$TrialBin <- factor(d$TrialBin)

# 剔 Ac/Am 前缀(其他任务): prefix = 剥离末尾 数字+后缀 后的字母前缀
d$prefix <- sub("\\d[FW]$", "", d$Treatment)
d_main <- d[!(d$prefix %in% c("Ac","Am")),]
d_exp  <- d_main[complete.cases(d_main$Age, d_main$Sex, d_main$Strain),]   # complete-case(有Age+Sex+Strain)
cat(sprintf("主分析(剔Ac/Am): %d trials, %d 鼠\n", nrow(d_main), nlevels(droplevels(d_main$MouseID))))
cat(sprintf("探索性(complete-case): %d trials, %d 鼠\n", nrow(d_exp), nlevels(droplevels(d_exp$MouseID))))

# 动态公式：因子水平<2 的项从公式剔除（避免 contrasts 错）
# 子集筛选后可能 ProbabilityCondition 只剩 1 水平(如纯 T9F→9:1)；探索性下可能 Restriction 纯 F
drop_terms_for <- function(dat, include_exp=FALSE){
    drop <- character(0)
    if (length(unique(dat$ProbabilityCondition)) < 2) drop <- c(drop, "ProbabilityCondition")
    if (length(unique(dat$PreviousOutcome))     < 2) drop <- c(drop, "PreviousOutcome")
    if (length(unique(dat$Restriction))         < 2) drop <- c(drop, "Restriction")
    if (length(unique(dat$SessionNum))          < 2) drop <- c(drop, "Session_s")
    if (include_exp){
        if (any(is.na(dat$Age)) || length(unique(na.omit(dat$Age))) < 2) drop <- c(drop, "Age")
        if (any(is.na(dat$Sex)) || length(unique(na.omit(dat$Sex)))  < 2) drop <- c(drop, "Sex")
        if (any(is.na(dat$Strain)) || length(unique(na.omit(dat$Strain))) < 2) drop <- c(drop, "Strain")
    }
    drop
}

# 构建固定效应字符串（按 drop_terms 动态拼；交互项依赖其主效应是否被 drop）
build_fe <- function(drop, poly=FALSE, include_exp=FALSE){
    terms <- "Trial_s"
    if (!"Session_s"           %in% drop) terms <- c(terms, "Session_s")
    if (!"ProbabilityCondition" %in% drop) terms <- c(terms, "ProbabilityCondition")
    if (!"PreviousOutcome"     %in% drop) terms <- c(terms, "PreviousOutcome")
    if (!"Restriction"         %in% drop) terms <- c(terms, "Restriction")
    if (poly) terms <- c(terms, "I(Trial_s^2)")
    if (include_exp){
        if (!"Age" %in% drop) terms <- c(terms, "Age", "Age:Trial_s")
        if (!"Sex" %in% drop && !"PreviousOutcome" %in% drop)
            terms <- c(terms, "Sex", "Sex:PreviousOutcome")
        else if (!"Sex" %in% drop)
            terms <- c(terms, "Sex")
        if (!"Strain" %in% drop) terms <- c(terms, "Strain")
    }
    paste(terms, collapse=" + ")
}

res_rows <- list()
# fit_one: 先试随机斜率 (Trial_s|MouseID); singularity/error → 回退 (1|MouseID)
fit_one <- function(fe_form, label, data, analysis){
    slope_form <- as.formula(paste0("ChoseHighProb ~ ", fe_form, " + (Trial_s|MouseID)"))
    intercept_form <- as.formula(paste0("ChoseHighProb ~ ", fe_form, " + (1|MouseID)"))
    t0 <- Sys.time()
    m <- tryCatch(
        glmer(slope_form, data=data, family=binomial(link="logit"),
              control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=1e5))),
        error=function(e) NULL)
    used_slope <- !is.null(m) && !isSingular(m)
    if (!used_slope){
        if (!is.null(m)) cat("  [回退] 随机斜率奇异/不收敛 → (1|MouseID)\n")
        m <- glmer(intercept_form, data=data, family=binomial(link="logit"),
                   control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=1e5)))
    }
    el <- as.numeric(difftime(Sys.time(), t0, units="secs"))
    fe <- fixef(m); vc <- vcov(m)
    for(nm in names(fe)){
        b <- fe[nm]; se <- sqrt(diag(vc))[nm]; z <- b/se; p <- 2*pnorm(-abs(z))
        or <- exp(b); lo <- exp(b-1.96*se); hi <- exp(b+1.96*se)
        res_rows[[length(res_rows)+1]] <<- data.frame(
            analysis=analysis, model=label, term=nm, beta=b, se=se, z=z, p=p,
            OR=or, OR_lo=lo, OR_hi=hi, AIC=AIC(m), BIC=BIC(m),
            logLik=as.numeric(logLik(m)), time_s=el,
            random_slope=used_slope, stringsAsFactors=FALSE)
    }
    list(m=m, used_slope=used_slope, AIC=AIC(m), logLik=as.numeric(logLik(m)), el=el)
}

# ====== 双分析循环 ======
analyses <- list(main=list(dat=d_main, exp=FALSE),
                 exp =list(dat=d_exp,  exp=TRUE))
fit_meta <- list()
for (an in names(analyses)){
    dat <- analyses[[an]]$dat; inc_exp <- analyses[[an]]$exp
    drop <- drop_terms_for(dat, include_exp=inc_exp)
    if (length(drop) > 0)
        cat(sprintf("\n注意[%s]: 以下因子水平<2，已从公式剔除: %s\n", an, paste(drop, collapse=", ")))
    cat(sprintf("\n========== %s 分析 (N=%d trials, %d 鼠) ==========\n",
        ifelse(an=="main","主","探索性"), nrow(dat), nlevels(droplevels(dat$MouseID))))

    cat("--- Model A: Trial_s 线性 ---\n")
    feA <- build_fe(drop, poly=FALSE, include_exp=inc_exp)
    rA <- fit_one(feA, "A_Trial_linear", dat, an)
    cat(sprintf("拟合 %.1fs  AIC=%.1f  logLik=%.1f  随机斜率=%s\n",
        rA$el, rA$AIC, rA$logLik, rA$used_slope))
    print(summary(rA$m)$coefficients, digits=4)

    cat("--- Model B: Trial_s + I(Trial_s^2) 多项式(非线性) ---\n")
    feB <- build_fe(drop, poly=TRUE, include_exp=inc_exp)
    rB <- fit_one(feB, "B_Trial_poly", dat, an)
    cat(sprintf("拟合 %.1fs  AIC=%.1f  logLik=%.1f  随机斜率=%s\n",
        rB$el, rB$AIC, rB$logLik, rB$used_slope))

    fit_meta[[an]] <- list(A=rA, B=rB, drop=drop)
}

res <- do.call(rbind, res_rows)
# 结果 CSV: 子集数据写到同名 .results.csv(紧挨数据文件), 全量默认仍 glme_results.csv
res_path <- if (dat_path == "/home/jyy/Documents/Beh_Ana_J/glme/glme_trial_data.csv") {
    "/home/jyy/Documents/Beh_Ana_J/glme/glme_results.csv"
} else {
    sub("\\.csv$", ".results.csv", dat_path)
}
write.csv(res, res_path, row.names=FALSE)
cat(sprintf("\n结果存 %s\n", res_path))

# ====== AIC 比较 + 摘要(按 analysis) ======
for (an in names(fit_meta)){
    fm <- fit_meta[[an]]
    drop <- fm$drop
    cat(sprintf("\n========== %s 分析 摘要 ==========\n", ifelse(an=="main","主","探索性")))
    cat(sprintf("AIC: Model A=%.1f  Model B=%.1f  ΔAIC=%.1f (%s)\n",
        fm$A$AIC, fm$B$AIC, fm$B$AIC-fm$A$AIC,
        ifelse(fm$B$AIC-fm$A$AIC < -2, "Model B 显著优", "Model A 不劣")))
    cat(sprintf("随机斜率: A=%s  B=%s\n", fm$A$used_slope, fm$B$used_slope))

    aa <- res[res$analysis==an & res$model=="A_Trial_linear",]
    cat("\n--- Model A 固定效应 β/p/OR ---\n")
    for(i in 1:nrow(aa)){
        cat(sprintf("%-28s β=%+.5f p=%.3g OR=%.4f [%.4f,%.4f]\n",
            aa$term[i], aa$beta[i], aa$p[i], aa$OR[i], aa$OR_lo[i], aa$OR_hi[i]))
    }
    cat(sprintf("鼠随机截距 SD=%.4f\n", attr(VarCorr(fm$A$m)$MouseID,"stddev")[[1]]))

    bb <- res[res$analysis==an & res$model=="B_Trial_poly" & grepl("Trial", res$term),]
    cat("\n--- Model B Trial 多项式系数(非线性检验) ---\n")
    for(i in 1:nrow(bb)){
        cat(sprintf("%-24s β=%+.5f p=%.3g OR=%.4f\n", bb$term[i], bb$beta[i], bb$p[i], bb$OR[i]))
    }
    q <- res[res$analysis==an & res$model=="B_Trial_poly" & res$term=="I(Trial_s^2)",]
    if (nrow(q)==1 && q$p < 0.05)
        cat("二次项 I(Trial_s^2) 显著(<0.05) => 学习曲线非线性(减速/饱和)\n")
    else
        cat("二次项不显著 => 线性已足够\n")
}

cat(sprintf("\nN: 主=%d trials/%d鼠  探索=%d trials/%d鼠\n",
    nrow(d_main), nlevels(droplevels(d_main$MouseID)),
    nrow(d_exp),  nlevels(droplevels(d_exp$MouseID))))
