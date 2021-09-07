#python RNNJF.py --out order_by_increasing_d0 --split 20000 --trial $t --order_by_feature d0
#python RNNJF.py --out order_by_decreasing_d0 --split 20000 --trial $t --order_by_feature d0 --reverse
#python RNNJF.py --out order_by_increasing_z0 --split 20000 --trial $t --order_by_feature z0
#python RNNJF.py --out order_by_decreasing_z0 --split 20000 --trial $t --order_by_feature z0 --reverse
#python RNNJF.py --out order_by_increasing_qp --split 20000 --trial $t --order_by_feature q/p
#python RNNJF.py --out order_by_decreasing_qp --split 20000 --trial $t --order_by_feature q/p --reverse
#python RNNJF.py --out order_by_increasing_r0 --split 20000 --trial $t --use_custom_order r0
#python RNNJF.py --out order_by_decreasing_r0 --split 20000 --trial $t --use_custom_order r0 --reverse
#python RNNJF.py --out order_by_increasing_t1 --split 20000 --trial $t --use_custom_order t1
#python RNNJF.py --out order_by_decreasing_t1 --split 20000 --trial $t --use_custom_order t1 --reverse


for t in 1 2 3 4 5
do
  python RNNJF.py --out baseline_20k --split 20000 --trial $t
  python RNNJF.py --out MinMax_d0 --split 20000 --trial $t --robust_scale z0 q/p
done

