# ============================================================
# eval_glme.R — GLME 模型评估：输出预测值 + 模型对象
# 用法: Rscript eval_glme.R  (环境变量 GLME_DATA 可覆盖数据路径)
# 输出:
#   glme_eval_preds.csv   — y_true, pred_fixed(边际), pred_full(含随机效应)
#   glme_eval_model.rds   — Model A 拟合对象
# ============================================================
suppressPackageStartupMessages({library(lme4)})

dat_path <- Sys.getenv("GLME_DATA", "/home/jyy/Documents/Beh_Ana_J/glme/glme_trial_data.csv")
out_dir  <- dirname(dat_path)
cat(sprintf("评估数据: %s\n", dat_path))

d <- read.csv(dat_path, stringsAsFactors=FALSE)
d$MouseID <- factor(d$MouseID)
d$ProbabilityCondition <- relevel(factor(d$ProbabilityCondition,
    levels=c("6:4","7:3","8:2","9:1")), ref="7:3")
d$PreviousOutcome <- factor(d$PreviousOutcome, levels=c(0,1))
d$Restriction <- factor(d$Restriction, levels=c("F","W"))
d$Age <- as.numeric(d$Age)
d$Sex  <- factor(d$Sex, levels=c(0,1))
d$Strain <- factor(d$Strain, levels=c("WT","Thy1"))
d$Trial_s  <- d$Trial / 1000
d$Session_s <- d$SessionNum / 10

# 剔 Ac/Am（主分析）
d$prefix <- sub("\\d[FW]$", "", d$Treatment)
d <- d[!(d$prefix %in% c("Ac","Am")),]
cat(sprintf("主分析: %d trials, %d 鼠\n", nrow(d), nlevels(droplevels(d$MouseID))))

# 动态公式
has_pc <- length(unique(d$ProbabilityCondition)) >= 2
has_po <- length(unique(d$PreviousOutcome)) >= 2
has_re <- length(unique(d$Restriction)) >= 2
has_ss <- length(unique(d$SessionNum)) >= 2

fe_terms <- "Trial_s"
if (has_ss) fe_terms <- c(fe_terms, "Session_s")
if (has_pc) fe_terms <- c(fe_terms, "ProbabilityCondition")
if (has_po) fe_terms <- c(fe_terms, "PreviousOutcome")
if (has_re) fe_terms <- c(fe_terms, "Restriction")
fe <- paste(fe_terms, collapse=" + ")

slope_form <- as.formula(paste0("ChoseHighProb ~ ", fe, " + (Trial_s|MouseID)"))
int_form   <- as.formula(paste0("ChoseHighProb ~ ", fe, " + (1|MouseID)"))
t0 <- Sys.time()
m <- tryCatch(
    glmer(slope_form, data=d, family=binomial,
          control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=1e5))),
    error=function(e) NULL)
used_slope <- !is.null(m) && !isSingular(m)
if (!used_slope) {
    cat("  [回退] 随机斜率奇异 → (1|MouseID)\n")
    m <- glmer(int_form, data=d, family=binomial,
               control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=1e5)))
}
el <- as.numeric(difftime(Sys.time(), t0, units="secs"))
cat(sprintf("拟合 %.1fs  AIC=%.1f  随机斜率=%s\n", el, AIC(m), used_slope))

pred_full  <- as.numeric(predict(m, type="response"))
pred_fixed <- as.numeric(predict(m, type="response", re.form=~0))

out <- data.frame(
    MouseID    = as.character(d$MouseID),
    Trial      = d$Trial,
    Trial_s    = d$Trial_s,
    y_true     = d$ChoseHighProb,
    pred_fixed = pred_fixed,
    pred_full  = pred_full,
    stringsAsFactors=FALSE
)

pred_path  <- file.path(out_dir, "glme_eval_preds.csv")
model_path <- file.path(out_dir, "glme_eval_model.rds")
write.csv(out, pred_path, row.names=FALSE)
saveRDS(m, model_path)
cat(sprintf("输出:\n  %s\n  %s\n", pred_path, model_path))
