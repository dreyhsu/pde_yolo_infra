import pandas as pd

data = [
    ["W14_WW_None_WeeklyMeeting", 14, "JQ", "SiM MP (52)", 4],
    ["W14_NK_1_PDE固定會議", 14, "NK1", "VMS(29)", 4],
    ["W14_NK_1_wahoo產品模型優化", 14, "NK1", "VMS(29)", 8],
    ["W14_NK_1_wahoo產品標註與訓練", 14, "NK1", "VMS(29)", 8],
    ["W14_JQ_SPI+AI改善率計算與分析", 14, "JQ", "SiM MP (52)", 8],
    ["W14_JQ_SPI+AI模型優化", 14, "JQ", "SiM MP (52)", 8],
    ["W15_WW_None_WeeklyMeeting", 15, "JQ", "SiM MP (52)", 4],
    ["W15_NK_1_PDE固定會議", 15, "NK1", "VMS(29)", 4],
    ["W15_NK_1_進線測試wahoo偵測流程", 15, "NK1", "VMS(29)", 8],
    ["W15_NK_1_標註RuckusP01包裝流程", 15, "NK1", "VMS(29)", 8],
    ["W15_JQ_SPI+AI改善率計算與分析", 15, "JQ", "SiM MP (52)", 8],
    ["W15_JQ_SPI+AI模型優化", 15, "JQ", "SiM MP (52)", 8],
    ["W16_WW_None_WeeklyMeeting", 16, "JQ", "SiM MP (52)", 4],
    ["W16_NK_1_PDE固定會議", 16, "NK1", "VMS(29)", 4],
    ["W16_NK_1_wahoo產品模型優化", 16, "NK1", "VMS(29)", 8],
    ["W16_NK_1_進線測試雙人作業流程", 16, "NK1", "VMS(29)", 8],
    ["W16_JQ_SPI+AI改善率計算與分析", 16, "JQ", "SiM MP (52)", 8],
    ["W16_JQ_SPI+AI輸出格式修改以符合報表所需", 16, "JQ", "SiM MP (52)", 8],
    ["W17_WW_None_WeeklyMeeting", 17, "JQ", "SiM MP (52)", 4],
    ["W17_NK_1_PDE固定會議", 17, "NK1", "VMS(29)", 4],
    ["W17_NK_1_wahoo產品包裝照片標註", 17, "NK1", "VMS(29)", 8],
    ["W17_NK_1_wahoo產品新模型訓練", 17, "NK1", "VMS(29)", 8],
    ["W17_JQ_SPI+AI改善率計算與分析", 17, "JQ", "SiM MP (52)", 8],
    ["W17_JQ_SPI+AI模型優化", 17, "JQ", "SiM MP (52)", 8],
]

df = pd.DataFrame(data, columns=["Name", "Weekly", "Sites", "BU(new)", "MH"])
df.to_excel("work_hours_2026_04.xlsx", index=False)
print("Successfully saved to work_hours_2026_04.xlsx")
