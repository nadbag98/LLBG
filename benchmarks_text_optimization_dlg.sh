# benchmarks for text stuff

# default trafo3:
# 32 x N
python benchmark_breaches.py name=deepleakage_trafo3_32_1 case=10_causal_lang_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=1 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
python benchmark_breaches.py name=deepleakage_trafo3_32_2 case=10_causal_lang_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=2 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
python benchmark_breaches.py name=deepleakage_trafo3_32_3 case=10_causal_lang_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=3 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
python benchmark_breaches.py name=deepleakage_trafo3_32_4 case=10_causal_lang_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=4 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
python benchmark_breaches.py name=deepleakage_trafo3_32_8 case=10_causal_lang_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=8 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
python benchmark_breaches.py name=deepleakage_trafo3_32_16 case=10_causal_lang_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=16 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
python benchmark_breaches.py name=deepleakage_trafo3_32_32 case=10_causal_lang_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=32 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
# oom python benchmark_breaches.py name=deepleakage_trafo3_32_64 case=10_causal_lang_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=64 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
# oom python benchmark_breaches.py name=deepleakage_trafo3_32_128 case=10_causal_lang_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=128 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
# M x 1
python benchmark_breaches.py name=deepleakage_trafo3_64_1 case=10_causal_lang_training attack=deepleakage case.data.shape=[64] case.user.num_data_points=1 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
python benchmark_breaches.py name=deepleakage_trafo3_128_1 case=10_causal_lang_training attack=deepleakage case.data.shape=[128] case.user.num_data_points=1 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
python benchmark_breaches.py name=deepleakage_trafo3_256_1 case=10_causal_lang_training attack=deepleakage case.data.shape=[256] case.user.num_data_points=1 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
python benchmark_breaches.py name=deepleakage_trafo3_512_1 case=10_causal_lang_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=1 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
python benchmark_breaches.py name=deepleakage_trafo3_1024_1 case=10_causal_lang_training attack=deepleakage case.data.shape=[1024] case.user.num_data_points=1 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
# 512 x N
python benchmark_breaches.py name=deepleakage_trafo3_512_1 case=10_causal_lang_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=1 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
python benchmark_breaches.py name=deepleakage_trafo3_512_2 case=10_causal_lang_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=2 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
python benchmark_breaches.py name=deepleakage_trafo3_512_3 case=10_causal_lang_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=3 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
# oom python benchmark_breaches.py name=deepleakage_trafo3_512_4 case=10_causal_lang_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=4 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True
# oom python benchmark_breaches.py name=deepleakage_trafo3_512_8 case=10_causal_lang_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=8 base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True

# bert-base
python benchmark_breaches.py name=deepleakage_bert_32_1 case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=1 case.model=bert-base-uncased base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
python benchmark_breaches.py name=deepleakage_bert_32_3 case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=3 case.model=bert-base-uncased base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
python benchmark_breaches.py name=deepleakage_bert_32_4 case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=4 case.model=bert-base-uncased base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
python benchmark_breaches.py name=deepleakage_bert_32_8 case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=8 case.model=bert-base-uncased base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
python benchmark_breaches.py name=deepleakage_bert_32_16 case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=16 case.model=bert-base-uncased base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
python benchmark_breaches.py name=deepleakage_bert_32_32 case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=32 case.model=bert-base-uncased base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
# oom python benchmark_breaches.py name=deepleakage_bert_32_64 case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=64 case.model=bert-base-uncased base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
# oom python benchmark_breaches.py name=deepleakage_bert_32_128 case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=128 case.model=bert-base-uncased base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
# M x 1
python benchmark_breaches.py name=deepleakage_bert_64_1 case=9_bert_training attack=deepleakage case.data.shape=[64] case.user.num_data_points=1 case.model=bert-base-uncased base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
python benchmark_breaches.py name=deepleakage_bert_128_1 case=9_bert_training attack=deepleakage case.data.shape=[128] case.user.num_data_points=1 case.model=bert-base-uncased base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
python benchmark_breaches.py name=deepleakage_bert_256_1 case=9_bert_training attack=deepleakage case.data.shape=[256] case.user.num_data_points=1 case.model=bert-base-uncased base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
python benchmark_breaches.py name=deepleakage_bert_512_1 case=9_bert_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=1 case.model=bert-base-uncased base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
python benchmark_breaches.py name=deepleakage_bert_1024_1 case=9_bert_training attack=deepleakage case.data.shape=[1024] case.user.num_data_points=1 case.model=bert-base-uncased base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
# 512 x N
python benchmark_breaches.py name=deepleakage_bert_512_1 case=9_bert_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=1 case.model=bert-base-uncased base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
python benchmark_breaches.py name=deepleakage_bert_512_2 case=9_bert_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=2 case.model=bert-base-uncased base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
# oom python benchmark_breaches.py name=deepleakage_bert_512_3 case=9_bert_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=3 case.model=bert-base-uncased base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
# oom python benchmark_breaches.py name=deepleakage_bert_512_4 case=9_bert_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=4 case.model=bert-base-uncased base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
# oom python benchmark_breaches.py name=deepleakage_bert_512_8 case=9_bert_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=8 case.model=bert-base-uncased base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100

 # strong-man their bert:
 python benchmark_breaches.py name=deepleakage_bert_sanity_32_1  case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=1 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True  attacker.optim.max_iterations=100
 python benchmark_breaches.py name=deepleakage_bert_sanity_32_3  case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=3 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True  attacker.optim.max_iterations=100
 python benchmark_breaches.py name=deepleakage_bert_sanity_32_4  case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=4 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True  attacker.optim.max_iterations=100
 python benchmark_breaches.py name=deepleakage_bert_sanity_32_8  case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=8 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True  attacker.optim.max_iterations=100
 python benchmark_breaches.py name=deepleakage_bert_sanity_32_16  case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=16 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True  attacker.optim.max_iterations=100
 python benchmark_breaches.py name=deepleakage_bert_sanity_32_32  case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=32 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=100
 # oom python benchmark_breaches.py name=deepleakage_bert_sanity_32_64  case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=64 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=100
 # oom python benchmark_breaches.py name=deepleakage_bert_sanity_32_128  case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=128 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=100
 # M x 1
 python benchmark_breaches.py name=deepleakage_bert_sanity_64_1 case=9_bert_training attack=deepleakage case.data.shape=[64] case.user.num_data_points=1 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=100
 python benchmark_breaches.py name=deepleakage_bert_sanity_128_1 case=9_bert_training attack=deepleakage case.data.shape=[128] case.user.num_data_points=1 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=100
 python benchmark_breaches.py name=deepleakage_bert_sanity_256_1 case=9_bert_training attack=deepleakage case.data.shape=[256] case.user.num_data_points=1 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=100
 python benchmark_breaches.py name=deepleakage_bert_sanity_512_1 case=9_bert_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=1 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=100
 python benchmark_breaches.py name=deepleakage_bert_sanity_1024_1 case=9_bert_training attack=deepleakage case.data.shape=[1024] case.user.num_data_points=1 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=100
 # 512 x N
 python benchmark_breaches.py name=deepleakage_bert_sanity_512_1  case=9_bert_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=1 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=100
 python benchmark_breaches.py name=deepleakage_bert_sanity_512_2  case=9_bert_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=2 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=100
 # oom python benchmark_breaches.py name=deepleakage_bert_sanity_512_3  case=9_bert_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=3 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
 # oom python benchmark_breaches.py name=deepleakage_bert_sanity_512_4  case=9_bert_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=4 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100
 # oom python benchmark_breaches.py name=deepleakage_bert_sanity_512_8  case=9_bert_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=8 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=100


 # strong-man their bert v2:
 python benchmark_breaches.py name=deepleakage_bert_sanity_iterate_longer_32_1  case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=1 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=1200
 python benchmark_breaches.py name=deepleakage_bert_sanity_iterate_longer_32_3  case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=3 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=1200
 python benchmark_breaches.py name=deepleakage_bert_sanity_iterate_longer_32_4  case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=4 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=1200
 python benchmark_breaches.py name=deepleakage_bert_sanity_iterate_longer_32_8  case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=8 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=1200
 python benchmark_breaches.py name=deepleakage_bert_sanity_iterate_longer_32_16  case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=16 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=1200
 python benchmark_breaches.py name=deepleakage_bert_sanity_iterate_longer_32_32  case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=32 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=1200
 # oom python benchmark_breaches.py name=deepleakage_bert_sanity_iterate_longer_32_64  case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=64 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=12000
 # oom python benchmark_breaches.py name=deepleakage_bert_sanity_iterate_longer_32_128  case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=128 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=12000
 # M x 1
 python benchmark_breaches.py name=deepleakage_bert_sanity_iterate_longer_64_1 case=9_bert_training attack=deepleakage case.data.shape=[64] case.user.num_data_points=1 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=1200
 python benchmark_breaches.py name=deepleakage_bert_sanity_iterate_longer_128_1 case=9_bert_training attack=deepleakage case.data.shape=[128] case.user.num_data_points=1 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=1200
 python benchmark_breaches.py name=deepleakage_bert_sanity_iterate_longer_256_1 case=9_bert_training attack=deepleakage case.data.shape=[256] case.user.num_data_points=1 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=1200
 python benchmark_breaches.py name=deepleakage_bert_sanity_iterate_longer_512_1 case=9_bert_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=1 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=1200
 python benchmark_breaches.py name=deepleakage_bert_sanity_iterate_longer_1024_1 case=9_bert_training attack=deepleakage case.data.shape=[1024] case.user.num_data_points=1 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=1200
 # 512 x N
 python benchmark_breaches.py name=deepleakage_bert_sanity_iterate_longer_512_1  case=9_bert_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=1 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=1200
 python benchmark_breaches.py name=deepleakage_bert_sanity_iterate_longer_512_2  case=9_bert_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=2 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=1200
 # oom python benchmark_breaches.py name=deepleakage_bert_sanity_iterate_longer_512_3  case=9_bert_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=3 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=12000
 # oom python benchmark_breaches.py name=deepleakage_bert_sanity_iterate_longer_512_4  case=9_bert_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=4 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=12000
 # oom python benchmark_breaches.py name=deepleakage_bert_sanity_iterate_longer_512_8  case=9_bert_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=8 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=12000

 # strong-man their bert v3: (using bert adam)
 python benchmark_breaches.py name=deepleakage_bert_sanity_bert_adam_32_1  case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=1 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=12000 attacker.optim.optimizer=bert-adam
 python benchmark_breaches.py name=deepleakage_bert_sanity_bert_adam_32_3  case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=3 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=12000 attacker.optim.optimizer=bert-adam
 python benchmark_breaches.py name=deepleakage_bert_sanity_bert_adam_32_4  case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=4 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=12000 attacker.optim.optimizer=bert-adam
 python benchmark_breaches.py name=deepleakage_bert_sanity_bert_adam_32_8  case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=8 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=12000 attacker.optim.optimizer=bert-adam
 python benchmark_breaches.py name=deepleakage_bert_sanity_bert_adam_32_16  case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=16 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=12000 attacker.optim.optimizer=bert-adam
 python benchmark_breaches.py name=deepleakage_bert_sanity_bert_adam_32_32  case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=32 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=12000 attacker.optim.optimizer=bert-adam
 # oom python benchmark_breaches.py name=deepleakage_bert_sanity_bert_adam_32_64  case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=64 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=12000 attacker.optim.optimizer=bert-adam
 # oom python benchmark_breaches.py name=deepleakage_bert_sanity_bert_adam_32_128  case=9_bert_training attack=deepleakage case.data.shape=[32] case.user.num_data_points=128 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=12000 attacker.optim.optimizer=bert-adam
 # M x 1
 python benchmark_breaches.py name=deepleakage_bert_sanity_bert_adam_64_1 case=9_bert_training attack=deepleakage case.data.shape=[64] case.user.num_data_points=1 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=12000 attacker.optim.optimizer=bert-adam
 python benchmark_breaches.py name=deepleakage_bert_sanity_bert_adam_128_1 case=9_bert_training attack=deepleakage case.data.shape=[128] case.user.num_data_points=1 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=12000 attacker.optim.optimizer=bert-adam
 python benchmark_breaches.py name=deepleakage_bert_sanity_bert_adam_256_1 attack=deepleakage case.data.shape=[256] case.user.num_data_points=1 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=12000 attacker.optim.optimizer=bert-adam
 python benchmark_breaches.py name=deepleakage_bert_sanity_bert_adam_512_1 case=9_bert_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=1 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=12000 attacker.optim.optimizer=bert-adam
 python benchmark_breaches.py name=deepleakage_bert_sanity_bert_adam_1024_1 case=9_bert_training attack=deepleakage case.data.shape=[1024] case.user.num_data_points=1 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=12000 attacker.optim.optimizer=bert-adam
 # 512 x N
 python benchmark_breaches.py name=deepleakage_bert_sanity_bert_adam_512_1  case=9_bert_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=1 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=12000 attacker.optim.optimizer=bert-adam
 python benchmark_breaches.py name=deepleakage_bert_sanity_bert_adam_512_2  case=9_bert_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=2 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True case.server.provide_public_buffers=True case.data.disable_mlm=True attacker.optim.max_iterations=12000 attacker.optim.optimizer=bert-adam
 # oom python benchmark_breaches.py name=deepleakage_bert_sanity_bert_adam_512_3  case=9_bert_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=3 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=12000 attacker.optim.optimizer=bert-adam
 # oom python benchmark_breaches.py name=deepleakage_bert_sanity_bert_adam_512_4  case=9_bert_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=4 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=12000 attacker.optim.optimizer=bert-adam
 # oom python benchmark_breaches.py name=deepleakage_bert_sanity_bert_adam_512_8  case=9_bert_training attack=deepleakage case.data.shape=[512] case.user.num_data_points=8 case.model=bert-sanity-check base_dir=/cmlscratch/jonas0/breaching/outputs save_reconstruction=True attacker.optim.max_iterations=12000 attacker.optim.optimizer=bert-adam
