use std::io::Write;

use candle_transformers::{models::llama as model, generation::LogitsProcessor};
use model::{Llama, LlamaConfig, Cache};

use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;

use anyhow::{bail, Error as E, Result};

#[cfg(feature = "accelerate")]
extern crate accelerate_src;


const EOS_TOKEN: &str = "</s>";
const DEFAULT_PROMPT: &str = "My favorite theorem is ";

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F32;

    let api = Api::new()?;
    let model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string();

    let revision = "main".to_string();
    let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

    let config_filename = api.get("config.json")?;
    let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    let config = config.into_config(false);

    let filenames = vec![api.get("model.safetensors")?];

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let cache = Cache::new(false, dtype, &config, &device)?;
    
    let llama = Llama::load(vb, &cache, &config)?;

    let tokenizer_filename = api.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    
    let eos_token_id = tokenizer.token_to_id(EOS_TOKEN);
    let prompt = DEFAULT_PROMPT;

    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    println!("starting the inference loop");
    print!("{prompt}");

    let seed = 299792458;
    let temperature = None; //Some(1.0);
    let top_p = None; //Some(0.9);
    let mut logits_processor = LogitsProcessor::new(seed, temperature, top_p);

    let mut index_pos = 0;
    let mut token_generated = 0;
    for index in 0..100 {
        let (context_size, context_index) = if cache.use_kv_cache && index > 0 {
            (1, index_pos)
        } else {
            (tokens.len(), 0)
        };
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];

        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = llama.forward(&input, context_index)?;
        let logits = logits.squeeze(0)?;
        index_pos += ctxt.len();
        
        let next_token = logits_processor.sample(&logits)?;
        token_generated += 1;
        tokens.push(next_token);

        if let Some(text) = tokenizer.id_to_token(next_token) {
            let text = text.replace('▁', " ").replace("<0x0A>", "\n");
            print!("{text}");
            std::io::stdout().flush()?;
        }
        if Some(next_token) == eos_token_id {
            break;
        }
    }

    Ok(())
}
