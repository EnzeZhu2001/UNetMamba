_base_ = [
    '../_base_/datasets/coco_vg_vqa.py',
    '../_base_/default_runtime.py',
]

# unetmamba settings
model = dict(
    type='BlipVQA',
    tokenizer=dict(type='BlipTokenizer', name_or_path='bert-base-uncased'),
    vision_backbone=dict(
        type='VisionTransformer',
        arch='b',
        img_size=480,
        patch_size=16,
        out_type='raw'),
    multimodal_backbone=dict(
        type='XBertEncoder',
        med_config=dict(
            architectures=['BertModel'],
            attention_probs_dropout_prob=0.1,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            hidden_size=768,
            initializer_range=0.02,
            intermediate_size=3072,
            layer_norm_eps=1e-12,
            max_position_embeddings=512,
            model_type='bert',
            num_attention_heads=12,
            num_hidden_layers=12,
            pad_token_id=0,
            add_type_embeddings=False,
            vocab_size=30524,
            encoder_width=768,
            add_cross_attention=True),
    ),
    head=dict(
        type='VQAGenerationHead',
        decoder=dict(
            type='XBertLMHeadDecoder',
            med_config=dict(
                architectures=['BertModel'],
                attention_probs_dropout_prob=0.1,
                hidden_act='gelu',
                hidden_dropout_prob=0.1,
                hidden_size=768,
                initializer_range=0.02,
                intermediate_size=3072,
                layer_norm_eps=1e-12,
                max_position_embeddings=512,
                model_type='bert',
                num_attention_heads=12,
                num_hidden_layers=12,
                pad_token_id=0,
                add_type_embeddings=False,
                vocab_size=30524,
                encoder_width=768,
                add_cross_attention=True),
        ),
        inference_method='rank',  # or 'generate'
        answer_list_path=
        'https://storage.googleapis.com/sfr-vision-language-research/datasets/answer_list.json',  # noqa: E501
    ),
)

# schedule settings
optimizer = dict(type='AdamW', lr=2e-5, weight_decay=0.05)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

param_scheduler = [dict(type='CosineAnnealingLR', by_epoch=True)]

train_cfg = dict(max_epochs=10, by_epoch=True)
test_cfg = dict()

# runtime settings
randomness = dict(seed=42)