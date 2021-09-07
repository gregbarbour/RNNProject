python RNNJF.py --out order_by_increasing_d0 --split 20000 --trial 1 --order_by_feature d0
python RNNJF.py --out order_by_increasing_d0 --split 20000 --trial 2 --order_by_feature d0
python RNNJF.py --out order_by_increasing_d0 --split 20000 --trial 3 --order_by_feature d0
python RNNJF.py --out order_by_decreasing_d0 --split 20000 --trial 1 --order_by_feature d0 --reverse
python RNNJF.py --out order_by_decreasing_d0 --split 20000 --trial 2 --order_by_feature d0 --reverse
python RNNJF.py --out order_by_decreasing_d0 --split 20000 --trial 3 --order_by_feature d0 --reverse

python RNNJF.py --out order_by_increasing_z0 --split 20000 --trial 1 --order_by_feature z0
python RNNJF.py --out order_by_increasing_z0 --split 20000 --trial 2 --order_by_feature z0
python RNNJF.py --out order_by_increasing_z0 --split 20000 --trial 3 --order_by_feature z0
python RNNJF.py --out order_by_decreasing_z0 --split 20000 --trial 1 --order_by_feature z0 --reverse
python RNNJF.py --out order_by_decreasing_z0 --split 20000 --trial 2 --order_by_feature z0 --reverse
python RNNJF.py --out order_by_decreasing_z0 --split 20000 --trial 3 --order_by_feature z0 --reverse

python RNNJF.py --out order_by_increasing_qp --split 20000 --trial 1 --order_by_feature q/p
python RNNJF.py --out order_by_increasing_qp --split 20000 --trial 2 --order_by_feature q/p
python RNNJF.py --out order_by_increasing_qp --split 20000 --trial 3 --order_by_feature q/p
python RNNJF.py --out order_by_decreasing_qp --split 20000 --trial 1 --order_by_feature q/p --reverse
python RNNJF.py --out order_by_decreasing_qp --split 20000 --trial 2 --order_by_feature q/p --reverse
python RNNJF.py --out order_by_decreasing_qp --split 20000 --trial 3 --order_by_feature q/p --reverse

python RNNJF.py --out order_by_increasing_r0 --split 20000 --trial 1 --use_custom_order r0
python RNNJF.py --out order_by_increasing_r0 --split 20000 --trial 2 --use_custom_order r0
python RNNJF.py --out order_by_increasing_r0 --split 20000 --trial 3 --use_custom_order r0
python RNNJF.py --out order_by_decreasing_r0 --split 20000 --trial 1 --use_custom_order r0 --reverse
python RNNJF.py --out order_by_decreasing_r0 --split 20000 --trial 2 --use_custom_order r0 --reverse
python RNNJF.py --out order_by_decreasing_r0 --split 20000 --trial 3 --use_custom_order r0 --reverse



