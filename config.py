#modelparameters
hidden_dim = 256
num_layers = 4
lr= 1e-5 #1e-5 was better
batch_size = 8
model_type = "transformer" # "transformer", "lstm" linear

MIN_LENGTH_TIMESERIES = 10
MAX_N_TIMESERIES = 10000
NAN_VALUE = -1.0

filepath = "data/CSI_Data_ALL_12062025_enriched.csv"
filter_features = ["site_cluster"]#["river", "site" ]
display_features = ["waterLevel", "temperature", "totalDissolvedSolids", "turbidity", "ph" ,"phosphate","nitrate", "ammonia"]#
input_features = ["long", "lat","recentRain", "estimatedWidth", "estimatedDepth", "waterFlow", "timestamp", "landUseWoodland", "landUseMoorlandOrHeath", "landUseUrbanResidential", "landUseIndustrialOrCommercial","landUseParklandOrGardens", "landUseGrasslandOrPasture" , "landUseAgriculture", "landUseTilledLand", "landUseOther", "pollutionEvidenceNone",'bankVegetationBareEarth', 'bankVegetationImpermeableSurface',
       'bankVegetationTreesOrShrubs', 'bankVegetationGrass',
       'bankVegetationOther','pollutionSourceNone',
       'pollutionSourceActiveOutfalls', 'pollutionSourceInactiveOutfalls',
       'pollutionSourceOutfallDiscolouration', 'pollutionSourceOutfallOdour',
       'pollutionSourceFarmRunoff', 'pollutionSourceGreyWater',
       'pollutionSourceCattleOrStock', 'pollutionSourceRiverBankCollapse',
       'pollutionSourceSoilRunoff', 'pollutionSourceRoadRunoff',
       'pollutionSourceOther', 
       'pollutionEvidenceSewageFungus', 'pollutionEvidenceOilySheen',
       'pollutionEvidenceSewageRelatedLitter',
       'pollutionEvidenceUnpleasantOdour',
       'pollutionEvidenceLitterOrFlyTipping', 'pollutionEvidenceFoam',
       'pollutionEvidenceSmotheringAlgae', 'pollutionEvidenceDeadFish',
       'pollutionEvidenceDeadInvertebrates', 'pollutionEvidenceOther',"OPCAT_ID", "MANCAT_ID","waterBodyType",'flowImpedanceNone',
       'flowImpedanceWeir', 'flowImpedanceTree', 'flowImpedanceDebrisDam',
       'flowImpedanceBridgeOrCulvert', 'flowImpedanceOther',
       "prcp_mm_d0","prcp_mm_d1","prcp_mm_d2","prcp_mm_d3","prcp_mm_d5","prcp_mm_d10"]#,

#determines which features are predicted by the model
intersting_columns = ["waterLevel", "temperature", "totalDissolvedSolids", "turbidity", "ph", "nitrate", "ammonia","phosphate"]#

#all features which are feed into the model
all_features =  input_features + intersting_columns
#target_features = ["pollutionEvidenceNone"]


#names
categorical_features_names = ["recentRain","waterFlow","nitrate","ammonia","waterLevel", "landUseWoodland", "landUseMoorlandOrHeath", "landUseUrbanResidential", "landUseIndustrialOrCommercial","landUseParklandOrGardens", "landUseGrasslandOrPasture" , "landUseAgriculture", "landUseTilledLand", "landUseOther","pollutionEvidenceNone",'bankVegetationBareEarth', 'bankVegetationImpermeableSurface',
       'bankVegetationTreesOrShrubs', 'bankVegetationGrass',
       'bankVegetationOther','pollutionSourceNone',
       'pollutionSourceActiveOutfalls', 'pollutionSourceInactiveOutfalls',
       'pollutionSourceOutfallDiscolouration', 'pollutionSourceOutfallOdour',
       'pollutionSourceFarmRunoff', 'pollutionSourceGreyWater',
       'pollutionSourceCattleOrStock', 'pollutionSourceRiverBankCollapse',
       'pollutionSourceSoilRunoff', 'pollutionSourceRoadRunoff',
       'pollutionSourceOther', 
       'pollutionEvidenceSewageFungus', 'pollutionEvidenceOilySheen',
       'pollutionEvidenceSewageRelatedLitter',
       'pollutionEvidenceUnpleasantOdour',
       'pollutionEvidenceLitterOrFlyTipping', 'pollutionEvidenceFoam',
       'pollutionEvidenceSmotheringAlgae', 'pollutionEvidenceDeadFish',
       'pollutionEvidenceDeadInvertebrates', 'pollutionEvidenceOther',"OPCAT_ID", "MANCAT_ID","waterBodyType",'flowImpedanceNone',
       'flowImpedanceWeir', 'flowImpedanceTree', 'flowImpedanceDebrisDam',
       'flowImpedanceBridgeOrCulvert', 'flowImpedanceOther',]# "OPCAT_ID", "MANCAT_ID",

attribute_names = all_features +["site_index"]

categorical_features_indices = [all_features.index(a) for a in categorical_features_names if a in all_features]
not_categorical_features_indices = [i for i in range(len(all_features)) if i not in categorical_features_indices]+[len(all_features)]
display_features_indices = [all_features.index(a) for a in display_features if a in all_features]
int_indices = [all_features.index(a) for a in intersting_columns if a in all_features]
loss_features_indices_cat = [a for a in int_indices if a in categorical_features_indices]
loss_features_indices_notcat = [a for a in int_indices if a in not_categorical_features_indices]

config = {
    "hidden_dim": hidden_dim,
    "num_layers": num_layers,
    "lr": lr,
    "batch_size": batch_size,
    "model_type": model_type,
    "filepath": filepath,
}
