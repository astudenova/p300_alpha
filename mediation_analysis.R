library("lme4")
library("mediation")

dat <- read.csv("~/forstatfile.csv",
                colClasses = c("numeric","numeric","numeric","numeric",
                               "numeric","numeric","numeric","numeric"))

dep_vars = list('attention', 'memory', 'executive')

# loop for latency
coeffs = matrix(, nrow = 3, ncol = 4)
for(i in 1:length(dep_vars)) { 
  
  dep_var = dep_vars[[i]]
  print(dep_var)
  fit.totaleffect=lm(paste(dep_var, '~erpL+age'),dat)
  print(summary(fit.totaleffect))
  coeffs[i,1] = fit.totaleffect$coefficients[[2]]
  
  fit.mediator=lm('envL~erpL',dat)
  print(summary(fit.mediator))
  coeffs[i,2] = fit.mediator$coefficients[[2]]
  
  fit.dv=lm(paste(dep_var,'~erpL+envL+age'),dat)
  print(summary(fit.dv))
  coeffs[i,3] = fit.dv$coefficients[[3]]
  coeffs[i,4] = fit.dv$coefficients[[2]]
  
  results = mediate(fit.mediator, fit.dv, treat='erpL', mediator='envL', covariates='age',boot=T)
  
  print(summary(results))
  
}

# loop for amplitude
coeffs = matrix(, nrow = 3, ncol = 4)
for(i in 1:length(dep_vars)) { 
  
  dep_var = dep_vars[[i]]
  print(dep_var)
  fit.totaleffect=lm(paste(dep_var, '~erpA+age'),dat)
  print(summary(fit.totaleffect))
  coeffs[i,1] = fit.totaleffect$coefficients[[2]]
  
  fit.mediator=lm('envA~erpA',dat)
  print(summary(fit.mediator))
  coeffs[i,2] = fit.mediator$coefficients[[2]]
  
  fit.dv=lm(paste(dep_var,'~erpA+envA+age'),dat)
  print(summary(fit.dv))
  coeffs[i,3] = fit.dv$coefficients[[3]]
  coeffs[i,4] = fit.dv$coefficients[[2]]
  
  results = mediate(fit.mediator, fit.dv, treat='erpA', mediator='envA', covariates='age',boot=T)
  
  print(summary(results))
  
}
