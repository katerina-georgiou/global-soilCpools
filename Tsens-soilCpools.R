#####################################################################################
###
### R script for climatological temperature sensitivities of soil carbon pools
### 
### Emergent temperature sensitivity of soil organic carbon driven by mineral associations
### Published in Nature Geoscience, 2024
### 
### Data sources: 
### CMIP6 ESMs: https://esgf-node.llnl.gov/search/cmip6/
### Biogeochemical Testbed: https://doi.org/10.5065/d6nc600w
### Data Product: https://doi.org/10.5281/zenodo.6539765
### All covariates are freely available in the references detailed in the manuscript, 
### and are also available from the corresponding author upon request.
###
### Contact: Katerina Georgiou (georgiou1@llnl.gov)
###
#####################################################################################

################ Importing packages ################ 

# importing packages
library(dplyr)
library(ggplot2)
library(plyr)
library(reshape2)
library(tidyr)
library(ggExtra)
library(gridExtra)
library(ggpubr)
library(RColorBrewer)
library(grid)
library(gtable)
library(ggrepel)

################ Colorbar functions ################ 

#### colorbar both sides
my_triangle_colourbar <- function(...) {
  guide <- guide_colourbar(...)
  class(guide) <- c("my_triangle_colourbar", class(guide))
  guide
}

guide_gengrob.my_triangle_colourbar <- function(...) {
  # First draw normal colourbar
  guide <- NextMethod()
  # Extract bar / colours
  is_bar <- grep("^bar$", guide$layout$name)
  bar <- guide$grobs[[is_bar]]
  extremes <- c(bar$raster[1], bar$raster[length(bar$raster)])
  # Extract size
  width  <- guide$widths[guide$layout$l[is_bar]]
  height <- guide$heights[guide$layout$t[is_bar]]
  short  <- min(convertUnit(width, "cm",  valueOnly = TRUE),
                convertUnit(height, "cm", valueOnly = TRUE))
  # Make space for triangles
  guide <- gtable_add_rows(guide, unit(short, "cm"),
                           guide$layout$t[is_bar] - 1)
  guide <- gtable_add_rows(guide, unit(short, "cm"),
                           guide$layout$t[is_bar])
  
  # Draw triangles
  top <- polygonGrob(
    x = unit(c(0, 0.5, 1), "npc"),
    y = unit(c(0, 1, 0), "npc"),
    gp = gpar(fill = extremes[1], col = NA)
  )
  bottom <- polygonGrob(
    x = unit(c(0, 0.5, 1), "npc"),
    y = unit(c(1, 0, 1), "npc"),
    gp = gpar(fill = extremes[2], col = NA)
  )
  # Add triangles to guide
  guide <- gtable_add_grob(
    guide, top, 
    t = guide$layout$t[is_bar] - 1,
    l = guide$layout$l[is_bar]
  )
  guide <- gtable_add_grob(
    guide, bottom,
    t = guide$layout$t[is_bar] + 1,
    l = guide$layout$l[is_bar]
  )
  
  return(guide)
}

#### colorbar top only
my_triangle_colourbar_top <- function(...) {
  guide <- guide_colourbar(...)
  class(guide) <- c("my_triangle_colourbar_top", class(guide))
  guide
}

guide_gengrob.my_triangle_colourbar_top <- function(...) {
  # First draw normal colourbar
  guide <- NextMethod()
  # Extract bar / colours
  is_bar <- grep("^bar$", guide$layout$name)
  bar <- guide$grobs[[is_bar]]
  extremes <- c(bar$raster[1], bar$raster[length(bar$raster)])
  # Extract size
  width  <- guide$widths[guide$layout$l[is_bar]]
  height <- guide$heights[guide$layout$t[is_bar]]
  short  <- min(convertUnit(width, "cm",  valueOnly = TRUE),
                convertUnit(height, "cm", valueOnly = TRUE))
  # Make space for triangles
  guide <- gtable_add_rows(guide, unit(short, "cm"),
                           guide$layout$t[is_bar] - 1)
  guide <- gtable_add_rows(guide, unit(short, "cm"),
                           guide$layout$t[is_bar])
  
  # Draw triangles
  top <- polygonGrob(
    x = unit(c(0, 0.5, 1), "npc"),
    y = unit(c(0, 1, 0), "npc"),
    gp = gpar(fill = extremes[1], col = NA)
  )
  
  # Add triangles to guide
  guide <- gtable_add_grob(
    guide, top, 
    t = guide$layout$t[is_bar] - 1,
    l = guide$layout$l[is_bar]
  )
  
  return(guide)
}

#

################ Importing and plotting global data ################ 

#### file names ####

setwd("~/")

# data & model output file names from the
# provided python script global-soilCpools.py
# using the data sources listed in the script header
dir_string <- c('ESM_output_Data-Product.csv',
                'ESM_output_ACCESS-ESM1-5.csv',
                'ESM_output_BCC-CSM2-MR.csv',
                'ESM_output_CESM2.csv',
                'ESM_output_CNRM-ESM2-1.csv',
                'ESM_output_E3SM-1-1-ECA.csv',
                'ESM_output_IPSL-CM6A-LR.csv',
                'ESM_output_MIROC-ES2L.csv',
                'ESM_output_MRI-ESM2-0.csv',
                'ESM_output_NorESM2.csv',
                'ESM_output_CASA-CNP.csv',
                'ESM_output_MIMICS.csv',
                'ESM_output_CORPSE.csv')

# data & model names
names_string <- c('Data-Product',
                  'ACCESS-ESM1-5',
                  'BCC-CSM2-MR',
                  'CESM2',
                  'CNRM-ESM2-1',
                  'E3SM-1-1-ECA',
                  'IPSL-CM6A-LR',
                  'MIROC-ES2L',
                  'MRI-ESM2-0',
                  'NorESM2',
                  'CASA-CNP',
                  'MIMICS',
                  'CORPSE')

length(dir_string)

#### data product ####

# importing data product
df <- read.table(dir_string[1] ,sep=",", header=TRUE, fill=TRUE)

# cleaning masked values
df <- subset(df, landcover > 0 & cSoil > 0 & cSoilSlow > 0 & npp > 0 & cSoil < 1000)

# cleaning change columns
df$change_cSoil <- NA
df$change_cSoilSlow <- NA
df$change_npp <- NA
df$change_pr <- NA
df$change_tas <- NA

# calculating unprotected C pool
df$cSoilFast <- df$cSoil-df$cSoilSlow

# matrix source
df$source <- names_string[1]

# checking
head(df)

# saving combined product
df_combined <- df

#### model outputs ####

for(i in 2:length(dir_string)) {
  print(names_string[i])
  
  # importing
  df <- read.table(dir_string[i] ,sep=",", header=TRUE, fill=TRUE)
  
  # cleaning masked values
  df <- subset(df, landcover > 0 & cSoil > 0 & cSoilSlow > 0 & npp > 0 & cSoil < 1000)
  
  # calculating unprotected C pool
  df$cSoilFast <- df$cSoil-df$cSoilSlow
  
  # matrix source
  df$source <- names_string[i]
  
  # saving combined product
  df_combined <- rbind(df_combined,df)
}

# cleaning
rm(df)

# adding vegetation names for all landcover
df_combined$Vegetation <- NA
df_combined$Vegetation <- ifelse(df_combined$landcover == 1, "Boreal Forest", df_combined$Vegetation)
df_combined$Vegetation <- ifelse(df_combined$landcover == 2, "Temperate Forest", df_combined$Vegetation)
df_combined$Vegetation <- ifelse(df_combined$landcover == 3, "Tropical Forest", df_combined$Vegetation)
df_combined$Vegetation <- ifelse(df_combined$landcover == 4, "Grassland", df_combined$Vegetation)
df_combined$Vegetation <- ifelse(df_combined$landcover == 5, "Shrubland", df_combined$Vegetation)
df_combined$Vegetation <- ifelse(df_combined$landcover == 6, "Savanna", df_combined$Vegetation)
df_combined$Vegetation <- ifelse(df_combined$landcover == 7, "Cropland", df_combined$Vegetation)
df_combined$Vegetation <- ifelse(df_combined$landcover == 8, "Tundra", df_combined$Vegetation)
df_combined$Vegetation <- ifelse(df_combined$landcover == 9, "Desert", df_combined$Vegetation)
df_combined$Vegetation <- ifelse(df_combined$landcover == 10, "Peatland", df_combined$Vegetation)

# checking
head(df_combined)

df_combined$source <- factor(df_combined$source, levels = c('Data-Product','ACCESS-ESM1-5','BCC-CSM2-MR',
                                                            'CESM2','CNRM-ESM2-1','E3SM-1-1-ECA','IPSL-CM6A-LR',
                                                            'MIROC-ES2L','MRI-ESM2-0','NorESM2',
                                                            'CASA-CNP','MIMICS','CORPSE'))

count(df_combined$source)

#### compiled data ####

df_combined <- subset(df_combined, select=c("source", "lats", "lons", "cSoil", "cSoilSlow", "cSoilFast", 
                                            "tas", "pr", "npp", "clay", "silt", "Vegetation", "grid_area"))

# df_combined is a compilation of the global data product, CMIP6 ESMs, and offline land models,
# with columns: source, lats, lons, cSoil, cSoilSlow, cSoilFast, tas, pr, npp, clay, silt, Vegetation, grid_area
# where source = global model or data product, lats = degrees latitude, lons = degrees longitude,
# cSoil = bulk soil C, cSoilSlow = protected soil C, cSoilFast = unprotected soil C (i.e., cSoil-cSoilSlow),
# tas = mean annual temperature, pr = precipitation, npp = net primary productivity, grid_area = area in m^2.

# calculating texture
df_combined$tex <- df_combined$clay+df_combined$silt

# ordering df_combined
df_combined$source <- factor(df_combined$source, levels = c('Data-Product', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CESM2',
                                                            'CNRM-ESM2-1', 'E3SM-1-1-ECA', 'IPSL-CM6A-LR', 'MIROC-ES2L',
                                                            'MRI-ESM2-0', 'NorESM2', 'CASA-CNP', 'MIMICS', 'CORPSE'))
# data & model names
names_string <- c('Data-Product', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CESM2',
                  'CNRM-ESM2-1', 'E3SM-1-1-ECA', 'IPSL-CM6A-LR', 'MIROC-ES2L',
                  'MRI-ESM2-0', 'NorESM2', 'CASA-CNP', 'MIMICS', 'CORPSE')

#### plotting global data ####

# plotting data product
i<- 1
sourcename <- names_string[i]
print(sourcename)

df_subset <- subset(df_combined, source == sourcename & tas > 0 & pr > 100 & Vegetation != "Peatland" & Vegetation != "Desert" & Vegetation != "Tundra")

## plotting total soil C vs. mean annual temperature colored by texture
ggplot(df_subset, aes(x=tas, y=(cSoil), color=(clay+silt))) + 
  geom_point(data = df_subset[seq(1, nrow(df_subset), 2), ], size=0.3) +
  scale_y_log10(limits = c(0.5,200)) + annotation_logticks(sides = "l") +
  labs(y = expression("Total soil C stocks (kg C m"^-{2}*")"), 
       x = expression("Mean annual temperature ("*~degree*C*")"),
       colour = expression("Clay + Silt (%)"))+
  theme_classic() + 
  scale_color_distiller(palette = "RdYlBu", limits=c(0,80), oob = scales::squish, guide = my_triangle_colourbar_top()) +
  geom_smooth(data = subset(df_subset, (clay+silt) < 20), method="nls", formula=y~SSasymp(x, Asym, R0, lrc), 
              color="black", fill="grey50", linetype="dashed", se=F, fullrange=F, linewidth=0.5) +
  geom_smooth(data = subset(df_subset, (clay+silt) > 70), method="nls", formula=y~SSasymp(x, Asym, R0, lrc), 
              color="black", fill="grey50", linetype="dashed", se=F, fullrange=F, linewidth=0.5) +
  theme(axis.text.x=element_text(size=12, colour="black"),
        axis.text.y=element_text(size=12, colour="black"),
        axis.title.x=element_text(size=14, colour="black"),
        axis.title.y=element_text(size=14, colour="black"),
        panel.background = element_rect(colour = "black", linewidth=0.8),
        legend.title.align=0.5,
        legend.text=element_text(size=10, colour="black"),
        legend.position = "none", legend.key.width = unit(0.3, "cm"), legend.key.height = unit(1, "cm"))

## plotting proportion protected vs. mean annual temperature colored by texture
ggplot(df_subset, aes(x=tas, y=(cSoilSlow/cSoil)*100, color=(clay+silt))) + 
  geom_point(data = df_subset[seq(1, nrow(df_subset), 2), ], size=0.3) +
  scale_y_continuous(
    minor_breaks = seq(0, 100, by = 5),
    breaks = seq(0, 100, by = 10), limits = c(0,100)) +
  labs(y = expression("Proportion of protected C (%)"), 
       x = expression("Mean annual temperature ("*~degree*C*")"),
       colour = "Clay + Silt (%)")+
  theme_classic() + 
  scale_color_distiller(palette = "RdYlBu", limits=c(0,80), oob = scales::squish, guide = my_triangle_colourbar_top()) +
  geom_smooth(data = subset(df_subset, (clay+silt) > 70),
              method = "lm", formula = y~x, color="black", linetype="dashed", linewidth=0.5) +
  geom_smooth(data = subset(df_subset, (clay+silt) < 20),
              method = "lm", formula = y~x, color="black", linetype="dashed", linewidth=0.5) +
  theme(axis.text.x=element_text(size=12, colour="black"),
        axis.text.y=element_text(size=12, colour="black"),
        axis.title.x=element_text(size=14, colour="black"),
        axis.title.y=element_text(size=14, colour="black"),
        panel.background = element_rect(colour = "black", linewidth=0.8),
        legend.title.align=0.5,
        legend.text = element_text(size=10, colour="black"),
        legend.position = "none", legend.key.width = unit(0.3, "cm"), legend.key.height = unit(1, "cm")) 

## plotting for protected C vs. mean annual temperature
ggplot(df_subset, aes(x=tas, y=(cSoilSlow))) + 
  geom_point(data = df_subset[seq(1, nrow(df_subset), 2), ], size=0.2, color="grey80") +
  scale_y_log10(limits = c(0.1,150)) + annotation_logticks(sides = "l") +
  labs(y = expression("Protected soil C stocks (kg C m"^-{2}*")"), 
       x = expression("Mean annual temperature ("*~degree*C*")"))+
  theme_classic() + 
  geom_smooth(data = df_subset, level=0.99, method = "lm", formula = y~x,
              color="chocolate4", fill="grey75", fullrange=F, span=2, linewidth=0.8, linetype = "dashed") +
  theme(axis.text.x=element_text(size=12, colour="black"),
        axis.text.y=element_text(size=12, colour="black"),
        axis.title.x=element_text(size=14, colour="black"),
        axis.title.y=element_text(size=14, colour="black"),
        panel.background = element_rect(colour = "black", size=0.8))

## plotting for unprotected C vs. mean annual temperature
ggplot(df_subset, aes(x=tas, y=(cSoilFast))) + 
  geom_point(data = df_subset[seq(1, nrow(df_subset), 2), ], size=0.2, color="grey80") + 
  scale_y_log10(limits = c(0.1,150)) + annotation_logticks(sides = "l") +
  labs(y = expression("Unprotected soil C stocks (kg C m"^-{2}*")"), 
       x = expression("Mean annual temperature ("*~degree*C*")"))+
  theme_classic() + 
  geom_smooth(data = df_subset, level=0.999, method = "lm", formula = y~x,
              color="green3", fill="grey75", fullrange=F, span=2, linewidth=0.8, linetype = "dashed") +
  theme(axis.text.x=element_text(size=12, colour="black"),
        axis.text.y=element_text(size=12, colour="black"),
        axis.title.x=element_text(size=14, colour="black"),
        axis.title.y=element_text(size=14, colour="black"),
        panel.background = element_rect(colour = "black", size=0.8))

## plotting proportion protected for all models colored by texture
ggplot(df_combined, aes(x=tas, y=(cSoilSlow/cSoil)*100, color=(clay+silt))) + 
  geom_point(data = subset(df_combined, pr > 100 & tas > 0 & Vegetation != "Peatland" & Vegetation != "Desert" & Vegetation != "Tundra")[seq(1, nrow(subset(df_combined, pr > 100 & tas > 0 & Vegetation != "Peatland" & Vegetation != "Desert" & Vegetation != "Tundra")), 8), ], size=0.1) +
  scale_y_continuous(
    minor_breaks = seq(0, 100, by = 5),
    breaks = seq(0, 100, by = 10), limits = c(0,100)) +
  labs(y = expression("Proportion of protected C (%)"), 
       x = expression("Mean annual temperature ("*~degree*C*")"),
       colour = "Clay + Silt (%)")+
  scale_color_distiller(palette = "RdYlBu", limits=c(0,80), oob = scales::squish, guide = my_triangle_colourbar_top()) +
  geom_smooth(data = subset(subset(df_combined, pr > 100 & npp > 10 & tas > 0 & Vegetation != "Peatland" & Vegetation != "Desert" & Vegetation != "Tundra"), (clay+silt) > 70),
              method = "lm", formula = y~x, color="darkred", linetype="dashed", linewidth=0.5) +
  geom_smooth(data = subset(subset(df_combined, pr > 100 & npp > 10 & tas > 0 & Vegetation != "Peatland" & Vegetation != "Desert" & Vegetation != "Tundra"), (clay+silt) < 20),
              method = "lm", formula = y~x, color="aquamarine4", linetype="dashed", linewidth=0.5) +
  theme_minimal() +
  theme(axis.text.x=element_text(size=12, colour="black"),
        axis.text.y=element_text(size=12, colour="black"),
        axis.title.x=element_text(size=14, colour="black"),
        axis.title.y=element_text(size=14, colour="black"),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        axis.ticks = element_line(colour = "black", size = 0.4),
        panel.background = element_rect(colour = "black", linewidth=0.8),
        legend.title.align=0.5,
        legend.text = element_text(size=10, colour="black"),
        legend.position = "right", legend.key.width = unit(0.3, "cm"), legend.key.height = unit(1, "cm")) + facet_wrap(vars(source))

## plotting proportion protected for all models over all temperatures
ggplot(df_combined, aes(x=tas, y=(cSoilSlow/cSoil)*100)) + 
  geom_vline(xintercept = 0, linetype='dashed', color="grey20") +
  geom_point(data = subset(df_combined, pr > 100 & npp > 10 & Vegetation != "Peatland" & Vegetation != "Desert")[seq(1, nrow(subset(df_combined, pr > 100 & npp > 10 & Vegetation != "Peatland" & Vegetation != "Desert")), 8), ], size=0.1, colour="grey10") + 
  scale_y_continuous(
    minor_breaks = seq(0, 100, by = 5),
    breaks = seq(0, 100, by = 10), limits = c(0,100)) +
  scale_x_continuous(breaks = seq(-15, 30, by = 5), limits = c(-15,30)) +
  labs(y = expression("Proportion of protected C (%)"), 
       x = expression("Mean annual temperature ("*~degree*C*")"),
       colour = "Clay + Silt (%)")+
  theme_minimal() +
  theme(axis.text.x=element_text(size=11, colour="black"),
        axis.text.y=element_text(size=11, colour="black"),
        axis.title.x=element_text(size=14, colour="black"),
        axis.title.y=element_text(size=14, colour="black"),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        panel.background = element_rect(colour = "black", linewidth=0.8),
        axis.ticks = element_line(colour = "black", size = 0.4),
        legend.title.align=0.5,
        legend.text = element_text(size=10, colour="black"),
        legend.position = "right", legend.key.width = unit(0.3, "cm"), legend.key.height = unit(1, "cm")) + facet_wrap(vars(source))

#

################ Climatological temperature sensitivities ################

#### calculations for all temperatures ####
#
# initializing t10 matrices (climatological temperature sensitivities for 10 degrees C)
cSoil <- matrix(NA, nrow = length(names_string), ncol = 3)
colnames(cSoil) <- c('value','low','high')
rownames(cSoil) <- names_string

cSoilSlow <- matrix(NA, nrow = length(names_string), ncol = 3)
colnames(cSoilSlow) <- c('value','low','high')
rownames(cSoilSlow) <- names_string

cSoilFast <- matrix(NA, nrow = length(names_string), ncol = 3)
colnames(cSoilFast) <- c('value','low','high')
rownames(cSoilFast) <- names_string

# looping through all datasets
for(i in 1:(length(names_string))) {
  sourcename <- names_string[i]
  print(sourcename)
  
  # note that the selected subset results in minor changes to the climatological temperature sensitivities, but conclusions remain the same
  df_subset <- subset(df_combined, source == sourcename & tas > 0 & pr > 100 & Vegetation != "Peatland" & Vegetation != "Desert" & Vegetation != "Tundra")
  
  # calculating T sensitivities
  lm_cSoil = lm(log(cSoil) ~ tas + npp + tex + pr, data = df_subset, na.action=na.omit)
  t10_cSoil = exp(coef(lm_cSoil)["tas"]*(-10))
  t10_cSoil_high = exp((confint(lm_cSoil, 'tas', level=0.95)[1])*(-10))
  t10_cSoil_low = exp((confint(lm_cSoil, 'tas', level=0.95)[2])*(-10))
  
  lm_cSoilSlow = lm(log(cSoilSlow) ~ tas + npp + tex + pr, data = df_subset, na.action=na.omit)
  t10_cSoilSlow = exp(coef(lm_cSoilSlow)["tas"]*(-10))
  t10_cSoilSlow_high = exp((confint(lm_cSoilSlow, 'tas', level=0.95)[1])*(-10))
  t10_cSoilSlow_low = exp((confint(lm_cSoilSlow, 'tas', level=0.95)[2])*(-10))
  
  lm_cSoilFast = lm(log(cSoilFast) ~ tas + npp + tex + pr, data = subset(df_subset, cSoilFast>0), na.action=na.omit)
  t10_cSoilFast = exp(coef(lm_cSoilFast)["tas"]*(-10))
  t10_cSoilFast_high = exp((confint(lm_cSoilFast, 'tas', level=0.95)[1])*(-10))
  t10_cSoilFast_low = exp((confint(lm_cSoilFast, 'tas', level=0.95)[2])*(-10))
  
  # saving T sensitivities
  cSoil[i,] <- c(t10_cSoil,t10_cSoil_low,t10_cSoil_high)
  cSoilSlow[i,] <- c(t10_cSoilSlow,t10_cSoilSlow_low,t10_cSoilSlow_high)
  cSoilFast[i,] <- c(t10_cSoilFast,t10_cSoilFast_low,t10_cSoilFast_high)
}

# checking and labeling 
cSoil <- data.frame(cSoil)
colnames(cSoil) <- c('value','low','high')
cSoil$variable <- "cSoil"
cSoil$source <- names_string

cSoilSlow <- data.frame(cSoilSlow)
colnames(cSoilSlow) <- c('value','low','high')
cSoilSlow$variable <- "cSoilSlow"
cSoilSlow$source <- names_string

cSoilFast <- data.frame(cSoilFast)
colnames(cSoilFast) <- c('value','low','high')
cSoilFast$variable <- "cSoilFast"
cSoilFast$source <- names_string

# combining
t10_control <- data.frame(rbind(cSoil,cSoilSlow,cSoilFast))
t10_control <- t10_control[, c('source','variable','value','low','high')]

# saving
t10_control$tempregime <- 'All'
t10_control_alltemp <- t10_control

#### calculations for cool temperatures ####
#
# initializing t10 matrices (climatological temperature sensitivities for 10 degrees C)
cSoil <- matrix(NA, nrow = length(names_string), ncol = 3)
colnames(cSoil) <- c('value','low','high')
rownames(cSoil) <- names_string

cSoilSlow <- matrix(NA, nrow = length(names_string), ncol = 3)
colnames(cSoilSlow) <- c('value','low','high')
rownames(cSoilSlow) <- names_string

cSoilFast <- matrix(NA, nrow = length(names_string), ncol = 3)
colnames(cSoilFast) <- c('value','low','high')
rownames(cSoilFast) <- names_string

# looping through all datasets
for(i in 1:(length(names_string))) {
  sourcename <- names_string[i]
  print(sourcename)
  
  df_subset <- subset(df_combined, source == sourcename & tas < 15 & tas > 0 & pr > 100 & Vegetation != "Peatland" & Vegetation != "Tundra")
  
  # calculating T sensitivities
  lm_cSoil = lm(log(cSoil) ~ tas + npp + tex + pr, data = df_subset, na.action=na.omit)
  t10_cSoil = exp(coef(lm_cSoil)["tas"]*(-10))
  t10_cSoil_high = exp((confint(lm_cSoil, 'tas', level=0.95)[1])*(-10))
  t10_cSoil_low = exp((confint(lm_cSoil, 'tas', level=0.95)[2])*(-10))
  
  lm_cSoilSlow = lm(log(cSoilSlow) ~ tas + npp + tex + pr, data = df_subset, na.action=na.omit)
  t10_cSoilSlow = exp(coef(lm_cSoilSlow)["tas"]*(-10))
  t10_cSoilSlow_high = exp((confint(lm_cSoilSlow, 'tas', level=0.95)[1])*(-10))
  t10_cSoilSlow_low = exp((confint(lm_cSoilSlow, 'tas', level=0.95)[2])*(-10))
  
  lm_cSoilFast = lm(log(cSoilFast) ~ tas + npp + tex + pr, data = subset(df_subset, cSoilFast>0), na.action=na.omit)
  t10_cSoilFast = exp(coef(lm_cSoilFast)["tas"]*(-10))
  t10_cSoilFast_high = exp((confint(lm_cSoilFast, 'tas', level=0.95)[1])*(-10))
  t10_cSoilFast_low = exp((confint(lm_cSoilFast, 'tas', level=0.95)[2])*(-10))
  
  # saving T sensitivities
  cSoil[i,] <- c(t10_cSoil,t10_cSoil_low,t10_cSoil_high)
  cSoilSlow[i,] <- c(t10_cSoilSlow,t10_cSoilSlow_low,t10_cSoilSlow_high)
  cSoilFast[i,] <- c(t10_cSoilFast,t10_cSoilFast_low,t10_cSoilFast_high)
  
}

# checking and labeling 
cSoil <- data.frame(cSoil)
colnames(cSoil) <- c('value','low','high')
cSoil$variable <- "cSoil"
cSoil$source <- names_string

cSoilSlow <- data.frame(cSoilSlow)
colnames(cSoilSlow) <- c('value','low','high')
cSoilSlow$variable <- "cSoilSlow"
cSoilSlow$source <- names_string

cSoilFast <- data.frame(cSoilFast)
colnames(cSoilFast) <- c('value','low','high')
cSoilFast$variable <- "cSoilFast"
cSoilFast$source <- names_string

# combining
t10_control <- data.frame(rbind(cSoil,cSoilSlow,cSoilFast))
t10_control <- t10_control[, c('source','variable','value','low','high')]

# saving
t10_control$tempregime <- 'Cool'
t10_control_0to15 <- t10_control

#### calculations for warm temperatures ####
#
# initializing t10 matrices (climatological temperature sensitivities for 10 degrees C)
cSoil <- matrix(NA, nrow = length(names_string), ncol = 3)
colnames(cSoil) <- c('value','low','high')
rownames(cSoil) <- names_string

cSoilSlow <- matrix(NA, nrow = length(names_string), ncol = 3)
colnames(cSoilSlow) <- c('value','low','high')
rownames(cSoilSlow) <- names_string

cSoilFast <- matrix(NA, nrow = length(names_string), ncol = 3)
colnames(cSoilFast) <- c('value','low','high')
rownames(cSoilFast) <- names_string

# looping through all datasets
for(i in 1:(length(names_string))) {
  sourcename <- names_string[i]
  print(sourcename)
  
  df_subset <- subset(df_combined, source == sourcename & tas >= 15 & pr > 100 & Vegetation != "Peatland" & Vegetation != "Desert" & Vegetation != "Tundra")
  
  # calculating T sensitivities
  lm_cSoil = lm(log(cSoil) ~ tas + npp + tex + pr, data = df_subset, na.action=na.omit)
  t10_cSoil = exp(coef(lm_cSoil)["tas"]*(-10))
  t10_cSoil_high = exp((confint(lm_cSoil, 'tas', level=0.95)[1])*(-10))
  t10_cSoil_low = exp((confint(lm_cSoil, 'tas', level=0.95)[2])*(-10))
  
  lm_cSoilSlow = lm(log(cSoilSlow) ~ tas + npp + tex + pr, data = df_subset, na.action=na.omit)
  t10_cSoilSlow = exp(coef(lm_cSoilSlow)["tas"]*(-10))
  t10_cSoilSlow_high = exp((confint(lm_cSoilSlow, 'tas', level=0.95)[1])*(-10))
  t10_cSoilSlow_low = exp((confint(lm_cSoilSlow, 'tas', level=0.95)[2])*(-10))
  
  lm_cSoilFast = lm(log(cSoilFast) ~ tas + npp + tex + pr, data = subset(df_subset, cSoilFast>0), na.action=na.omit)
  t10_cSoilFast = exp(coef(lm_cSoilFast)["tas"]*(-10))
  t10_cSoilFast_high = exp((confint(lm_cSoilFast, 'tas', level=0.95)[1])*(-10))
  t10_cSoilFast_low = exp((confint(lm_cSoilFast, 'tas', level=0.95)[2])*(-10))
  
  # saving T sensitivities
  cSoil[i,] <- c(t10_cSoil,t10_cSoil_low,t10_cSoil_high)
  cSoilSlow[i,] <- c(t10_cSoilSlow,t10_cSoilSlow_low,t10_cSoilSlow_high)
  cSoilFast[i,] <- c(t10_cSoilFast,t10_cSoilFast_low,t10_cSoilFast_high)
}

# checking and labeling 
cSoil <- data.frame(cSoil)
colnames(cSoil) <- c('value','low','high')
cSoil$variable <- "cSoil"
cSoil$source <- names_string

cSoilSlow <- data.frame(cSoilSlow)
colnames(cSoilSlow) <- c('value','low','high')
cSoilSlow$variable <- "cSoilSlow"
cSoilSlow$source <- names_string

cSoilFast <- data.frame(cSoilFast)
colnames(cSoilFast) <- c('value','low','high')
cSoilFast$variable <- "cSoilFast"
cSoilFast$source <- names_string

# combining
t10_control <- data.frame(rbind(cSoil,cSoilSlow,cSoilFast))
t10_control <- t10_control[, c('source','variable','value','low','high')]

# saving
t10_control$tempregime <- 'Warm'
t10_control_15plus <- t10_control

# cleaning
rm(cSoil,cSoilFast,cSoilSlow)
rm(lm_cSoil,lm_cSoilFast,lm_cSoilSlow)
rm(t10_cSoil,t10_cSoil_high,t10_cSoil_low,
   t10_cSoilSlow,t10_cSoilSlow_low,t10_cSoilSlow_high,
   t10_cSoilFast,t10_cSoilFast_high,t10_cSoilFast_low)
rm(t10_control)

#### combined output ####

# saving climatological temperature sensitivities for each regime
t10_control_all <- rbind(t10_control_alltemp, t10_control_0to15,t10_control_15plus)
rm(t10_control_alltemp, t10_control_0to15,t10_control_15plus)

t10_control_all$source <- factor(t10_control_all$source, levels = c('Data-Product', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CESM2',
                                                                    'CNRM-ESM2-1', 'E3SM-1-1-ECA', 'IPSL-CM6A-LR', 'MIROC-ES2L',
                                                                    'MRI-ESM2-0', 'NorESM2', 'CASA-CNP', 'MIMICS', 'CORPSE'))

t10_control_all$tempregime <- factor(t10_control_all$tempregime, levels = c('Cool', 'All', 'Warm'))

#### plotting climatological T sensitivities ####

## plotting data product climatological T sensitivities with temperature regimes
ggplot(subset(t10_control_all, variable != "cSoil" & source == "Data-Product"), 
       aes(x=tempregime, y=value, color=variable)) +
  geom_point(size=2.5) +
  geom_errorbar(aes(ymin=low, ymax=high), width=.2) + 
  labs(y = expression("Proportional decline in carbon (10"*~degree*C*")"), x = "")+
  scale_color_manual(values=c('green3', 'chocolate4')) +
  scale_y_continuous(
    minor_breaks = seq(1.0, 3.0, by = 0.1),
    breaks = seq(1.0, 3.0, by = 0.5), limits = c(0.9, 3.2)) +
  theme_classic() +
  theme(axis.text.x=element_text(size=11, colour="black", angle=90, vjust = 0.5, hjust=1),
        axis.text.y=element_text(size=12, colour="black"),
        axis.title.x=element_text(size=16, colour="black"),
        axis.title.y=element_text(size=14, colour="black"),
        legend.title=element_blank(),
        legend.text=element_text(size=12),
        panel.background = element_rect(colour = "black", size=0.8)) +
  geom_hline(yintercept = 1, linetype='dotted', color="grey75")

## plotting temperature sensitivities for all models
ggplot() + 
  geom_hline(yintercept = 1, linetype='dotted', color="grey75") +
  geom_point(data=subset(t10_control_all, tempregime == "All" & variable == "cSoilFast"), 
             mapping=aes(x=source, y=value), alpha=0.9, color="green3", size=3.5) +
  geom_point(data=subset(t10_control_all, tempregime == "All" & variable == "cSoilSlow"), 
             mapping=aes(x=source, y=value), alpha=0.9, color="chocolate4", size=3.5) +
  geom_point(data=subset(t10_control_all, tempregime == "All" & variable == "cSoil"), 
             mapping=aes(x=source, y=value), color="grey25", size=3, shape=5) +
  labs(y = expression("Proportional decline in carbon (10"*~degree*C*")"), x = "")+
  scale_y_continuous(
    minor_breaks = seq(0.5, 5, by = 0.1),
    breaks = seq(0.5, 3.5, by = 0.5), limits = c(0.5, 3.5)) +
  theme_classic() +
  theme(axis.text.x=element_text(size=11, colour="black", angle=90, vjust = 0.5, hjust=1),
        axis.text.y=element_text(size=12, colour="black"),
        axis.title.x=element_text(size=16, colour="black"),
        axis.title.y=element_text(size=14, colour="black"),
        legend.title=element_blank(),
        legend.text=element_text(size=12),
        legend.position = "none",
        panel.background = element_rect(colour = "black", size=0.8)) +
  geom_hline(yintercept = 1, linetype='dotted', color="grey75") +
  geom_vline(xintercept = 1.5, linetype='solid', color="black")

## plotting temperature sensitivities for all regimes
ggplot() + 
  geom_hline(yintercept = 1, linetype='dotted', color="grey75") +
  geom_point(data=subset(t10_control_all, variable == "cSoilFast"), 
             mapping=aes(x=tempregime, y=value), alpha=0.9, color="green3", size=3.5) +
  geom_point(data=subset(t10_control_all, variable == "cSoilSlow"), 
             mapping=aes(x=tempregime, y=value), alpha=0.9, color="chocolate4", size=3.5) +
  geom_point(data=subset(t10_control_all, variable == "cSoil"), 
             mapping=aes(x=tempregime, y=value), color="grey25", size=3, shape=5) +
  labs(y = expression("Prop. decline in carbon (10"*~degree*C*")"), x = "")+
  scale_y_continuous(
    minor_breaks = seq(0.5, 5, by = 0.1),
    breaks = seq(0.5, 5, by = 0.5), limits = c(0.5, 5)) +
  theme_minimal() +
  theme(axis.text.x=element_text(size=11, colour="black", angle=90, vjust = 0.5, hjust=1),
        axis.text.y=element_text(size=12, colour="black"),
        axis.title.x=element_text(size=16, colour="black"),
        axis.title.y=element_text(size=14, colour="black"),
        legend.title=element_blank(),
        legend.text=element_text(size=12),
        legend.position = "none",
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        axis.ticks = element_line(colour = "black", size = 0.4),
        panel.background = element_rect(colour = "black", size=0.8)) +
  geom_hline(yintercept = 1, linetype='dotted', color="grey75") + facet_wrap(vars(source))

#

#####################################################################################
