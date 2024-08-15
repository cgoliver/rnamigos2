POCKETS=(
	"5NDV_1_PAR_3405"
	"5NDV_5_PAR_3403"
	"1ZZ5_A_CNY_43"
	"6HIY_CA_SPM_735"
	"5KVJ_A_ARG_101"
	"5NDV_1_PAR_3414"
	"1BYJ_A_GE3_30"
	"5OBM_5_LLL_4164"
	"6XRQ_A_V8A_103"
	"6QIS_B_J48_101"
	"6QIS_H_J48_101"
	"6OM6_1_KKL_3301"
	"4V78_BA_FME_3001"
	"7OA3_B_GTP_102"
	"5OBM_6_LLL_2169"
	"6QIS_E_J48_102"
	"4V76_BA_FME_3001"
	"4YBB_DA_1PE_3185"
	"5NDV_1_PAR_3433"
	"7NSI_BA_SPM_1793"
	"6DB8_R_G4A_101"
	"2KXM_A_RIO_101"
	"4K31_C_AM2_102"
	"4V72_BA_FME_3001"
	"6JJI_B_POH_104"
	"1I97_A_TAC_2005"
	"4V6Y_BA_FME_3001"
	"4K31_B_AM2_101"
	"2MXS_A_PAR_101"
	"6LAU_A_GTP_103"
	"6AZ3_1_PAR_1802"
	"5LWJ_A_GTP_101"
	"6QIT_C_J48_101"
	"5NDG_5_GET_3851"
	"2L8H_A_L8H_2"
	"6QIT_D_J48_101"
	"1F1T_A_ROS_101"
	"6E8S_B_SPM_107"
	"6BSJ_R_TAM_101"
	"4V70_BA_FME_3001"
	"5NDV_1_PAR_3404"
	"5NDW_5_8UZ_3852"
	"6E82_A_TFX_102"
	"5V3F_A_74G_104"
	"6GZR_A_FH8_101"
	"5OBM_6_LLL_2176"
	"6S0X_A_ERY_3001"
	"4V53_AA_LLL_2062"
	"7N2V_16_SCM_1602"
	"2G5K_B_AM2_101"
	"5OBM_5_LLL_4167"
	"2F4V_A_D2C_1636"
	"2ET5_B_RIO_52"
	"3C44_B_PAR_24"
	"5NDW_1_8UZ_3886"
	"1TOB_A_TOA_28"
	"5OBM_1_LLL_3997"
	"4DR4_A_PAR_1612"
	"6CZR_1A_FSD_3801"
	"5DGE_1_SPS_4113"
	"2QEX_0_NEG_8823"
	"1NBK_A_GND_35"
	"6AZ3_7_PAR_201"
	"6CHR_A_SPM_741"
	"4X4N_B_5GP_102"
	"7TDB_A_GMI_101"
	"5BTP_B_AMZ_108"
	"6FZ0_A_SAM_104"
	"6VUI_A_PRF_101"
)


source .venv/bin/activate

for p in ${POCKETS[@]}; do
	echo "$p"
	IFS="_" read -r -a pocket_info <<< "$p"
	pdbid=${pocket_info[0]:l}
	chain=${pocket_info[1]}
	lig=${pocket_info[2]}
	pos=${pocket_info[3]}
	# ChimeraX --nogui  --cmd="open data/test_set_pdbs/${pdbid}.cif; save data/lig_mol2s/${pdbid}_${lig}.mol2 format mol2 atoms /${chain}:${pos}; exit"
	python scripts_fig/cam.py \
				--pocket-id ${pdbid:u}_${chain}_${lig}_${pos} \
				--mol2-path data/lig_mol2s/${pdbid}_${lig}.mol2 \
				--outpath-pocket outputs/CAM/${pdbid}_${chain}_${lig}_pocketcam.cxc \
				--outpath-lig outputs/CAM/${pdbid}_${chain}_${lig}_ligcam.cxc
	#ChimeraX --cmd="open data/test_set_pdbs/${pdbid}.cif; \
	#		open outputs/${pdbid}_${chain}_${lig}_pocketcam.cxc; \
	#		open outputs/${pdbid}_${chain}_${lig}_ligcam.cxc;
	#		"
done
