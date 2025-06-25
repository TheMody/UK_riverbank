MIN_LENGTH_TIMESERIES = 10
MAX_N_TIMESERIES = 10000
NAN_VALUE = -1.0

filter_features = ["site_cluster"]#["river", "site" ]
display_features = ["waterLevel", "temperature", "totalDissolvedSolids", "turbidity", "ph" ,"phosphate","nitrate", "ammonia"]#
input_features = ["OPCAT_ID", "MANCAT_ID","long", "lat","recentRain", "estimatedWidth", "estimatedDepth", "waterFlow", "timestamp", "landUseWoodland", "landUseMoorlandOrHeath", "landUseUrbanResidential", "landUseIndustrialOrCommercial","landUseParklandOrGardens", "landUseGrasslandOrPasture" , "landUseAgriculture", "landUseTilledLand", "landUseOther"]#
intersting_columns = ["waterLevel", "temperature", "totalDissolvedSolids", "turbidity", "ph", "nitrate", "ammonia","phosphate"]#
all_features =  input_features + intersting_columns
target_features = ["pollutionEvidenceNone"]

categorical_features_names = ["OPCAT_ID", "MANCAT_ID","recentRain","waterFlow","nitrate","ammonia","waterLevel", "landUseWoodland", "landUseMoorlandOrHeath", "landUseUrbanResidential", "landUseIndustrialOrCommercial","landUseParklandOrGardens", "landUseGrasslandOrPasture" , "landUseAgriculture", "landUseTilledLand", "landUseOther","pollutionEvidenceNone" ]# 

attribute_names = all_features +["site_index"]

categorical_features_indices = [all_features.index(a) for a in categorical_features_names if a in all_features]
not_categorical_features_indices = [i for i in range(len(all_features)) if i not in categorical_features_indices]+[len(all_features)]
display_features_indices = [all_features.index(a) for a in display_features if a in all_features]
