import json,sys
c=json.load(open(sys.argv[1]))["text_config"]
lt=c["layer_types"]; th=c["layer_rope_theta"]
print("sliding",lt.count("sliding_attention"),"full",lt.count("full_attention"))
print("full idx",[i for i,t in enumerate(lt) if t=="full_attention"])
print("full iff i%4==3:", all((t=="full_attention")==(i%4==3) for i,t in enumerate(lt)))
print("theta zero iff full:", all((th[i]==0)==(lt[i]=="full_attention") for i in range(52)))
print("tie",c["tie_word_embeddings"], "has keys", [k for k in ("use_qk_norm","normalize_tok_embeddings","output_soft_cap_temp","scale_query_by") if k in c])
