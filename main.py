from typing import Union, List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    PreTrainedTokenizerFast,
)

from CodonTransformer.CodonData import get_merged_seq
from CodonTransformer.CodonPrediction import predict_dna_sequence
from CodonTransformer.CodonUtils import (
    AMINO_ACID_TO_INDEX,
    INDEX2TOKEN,
    DNASequencePrediction,
)
from CodonTransformer.CodonJupyter import format_model_output


from CodonTransformer.CodonPrediction import validate_and_convert_organism
from CodonTransformer.CodonPrediction import tokenize
from CodonTransformer.CodonPrediction import load_tokenizer, load_model
from CodonTransformer.CodonPrediction import sample_non_deterministic

def predict_dna_sequence_annotated(
    protein: str,
    organism: Union[int, str],
    device: torch.device,
    tokenizer: Union[str, PreTrainedTokenizerFast] = None,
    model: Union[str, torch.nn.Module] = None,
    attention_type: str = "original_full",
    deterministic: bool = True,
    temperature: float = 0.2,
    top_p: float = 0.95,
    num_sequences: int = 1,
    match_protein: bool = False,
) -> Union[DNASequencePrediction, List[DNASequencePrediction]]:
    print(f"Predicting DNA sequence for protein: {protein}, organism: {organism}")

    if not protein:
        raise ValueError("Protein sequence cannot be empty.")

    if not isinstance(temperature, (float, int)) or temperature <= 0:
        raise ValueError("Temperature must be a positive float.")

    if not isinstance(top_p, (float, int)) or not 0 < top_p <= 1.0:
        raise ValueError("top_p must be a float between 0 and 1.")

    if not isinstance(num_sequences, int) or num_sequences < 1:
        raise ValueError("num_sequences must be a positive integer.")

    if deterministic and num_sequences > 1:
        raise ValueError(
            "Multiple sequences can only be generated in non-deterministic mode."
        )

    # Load tokenizer
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        tokenizer = load_tokenizer(tokenizer)

    # Load model
    if not isinstance(model, torch.nn.Module):
        model = load_model(model, device=device, attention_type=attention_type)
    else:
        model.eval()
        model.bert.set_attention_type(attention_type)
        model.to(device)

    # Validate organism and convert to organism_id and organism_name
    organism_id, organism_name = validate_and_convert_organism(organism)

    # Inference loop
    with torch.no_grad():
        # Tokenize the input sequence
        merged_seq = get_merged_seq(protein=protein, dna="")
        input_dict = {
            "idx": 0,  # sample index
            "codons": merged_seq,
            "organism": organism_id,
        }
        tokenized_input = tokenize([input_dict], tokenizer=tokenizer).to(device)

        # Get the model predictions
        output_dict = model(**tokenized_input, return_dict=True)
        logits = output_dict.logits.detach().cpu()
        logits = logits[:, 1:-1, :]  # Remove [CLS] and [SEP] tokens

        # Mask the logits of codons that do not correspond to the input protein sequence
        if match_protein:
            possible_tokens_per_position = [
                AMINO_ACID_TO_INDEX[token[0]] for token in merged_seq.split(" ")
            ]
            mask = torch.full_like(logits, float("-inf"))

            for pos, possible_tokens in enumerate(possible_tokens_per_position):
                mask[:, pos, possible_tokens] = 0

            logits = mask + logits

        predictions = []
        for _ in range(num_sequences):
            # Decode the predicted DNA sequence from the model output
            if deterministic:
                predicted_indices = logits.argmax(dim=-1).squeeze().tolist()
            else:
                predicted_indices = sample_non_deterministic(
                    logits=logits, temperature=temperature, top_p=top_p
                )

            predicted_dna = list(map(INDEX2TOKEN.__getitem__, predicted_indices))
            predicted_dna = (
                "".join([token[-3:] for token in predicted_dna]).strip().upper()
            )

            predictions.append(
                DNASequencePrediction(
                    organism=organism_name,
                    protein=protein,
                    processed_input=merged_seq,
                    predicted_dna=predicted_dna,
                )
            )

    return predictions[0] if num_sequences == 1 else predictions

def main():
    print ("Starting Codon Transformer prediction...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
    model = AutoModelForMaskedLM.from_pretrained(
        "adibvafa/CodonTransformer",
        trust_remote_code=True
    ).to(device)

    # Set your input data
    protein = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGG"
    organism = "Escherichia coli general"

    result = predict_dna_sequence_annotated(
        protein=protein,
        organism=organism,
        device=device,
        tokenizer=tokenizer,
        model=model
    )
    print(result)




if __name__ == "__main__":
    main()