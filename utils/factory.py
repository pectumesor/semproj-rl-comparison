from models import (MLPObservationEmbeddings, CNNObservationEmbeddings,
                     MLPBackbone, SimpleLSTM, GuassianPolicyHead, ValueNet, PPOAgent,
                      SquashedGaussianPolicyHead, DoubleQNet, SACAgent)
from omegaconf import DictConfig


def create_observation_model(type: str, ray_dim: int, proprio_dim:int,  cfg: DictConfig):

    if type == "MLP":
        return MLPObservationEmbeddings(
            input_dim=ray_dim + proprio_dim,
            hidden_sizes=cfg.model.obs_embed_hidden_sizes,
            feature_dim=cfg.model.obs_embed_hidden_sizes[-1]
            )
    elif type == "CNN":

        return CNNObservationEmbeddings(ray_channels=ray_dim,
                                        cnn_out_channels=cfg.model.cnn_output_channels,
                                        proprio_dim=proprio_dim,
                                        proprio_hidden_sizes=cfg.model.obs_embed_hidden_sizes,
                                        feature_dim=cfg.model.obs_embed_hidden_sizes[-1],
                                        )
    
def create_backbone_model(type: str, cfg: DictConfig):

    if type == "MLP":

        return MLPBackbone(input_dim=cfg.model.obs_embed_hidden_sizes[-1],
                           hidden_sizes=cfg.model.backbone_hidden_sizes,
                           output_dim=cfg.model.backbone_hidden_sizes[-1])
    elif type == "LSTM":

        return SimpleLSTM(input_dim=cfg.model.obs_embed_hidden_sizes[-1],
                          feature_dim= cfg.model.lstm_backbone_feature_dim,
                          num_layers=cfg.model.lstm_num_layers)
    
def create_ppo_agent(observation_type: str, backbone_type: str, ray_dim:int, proprio_dim:int, cfg: DictConfig):

    observation_model = create_observation_model(type=observation_type,
                                                 ray_dim=ray_dim, 
                                                 proprio_dim= proprio_dim,
                                                 cfg=cfg
                                                )
    
    backbone_model = create_backbone_model(type=backbone_type,
                                           cfg=cfg)
    
    actor = GuassianPolicyHead(backbone_dim=cfg.model.backbone_hidden_sizes[-1],
                                actions_dim=cfg.env.act_dim,
                                hidden_sizes=cfg.model.policy_hidden_sizes)
        
    critic = ValueNet(backbone_dim=cfg.model.backbone_hidden_sizes[-1],
                     hidden_sizes=cfg.model.value_hidden_sizes)
    
    return PPOAgent(obs_embed_model= observation_model,
                    backbone_model= backbone_model,
                    actor=actor, critic=critic,
                    action_low=cfg.env.action_low, action_high=cfg.env.action_high)

def create_sac_agent(observation_type: str, backbone_type: str, ray_dim:int, proprio_dim:int, cfg: DictConfig):
     
    observation_model = create_observation_model(type=observation_type,
                                                 ray_dim=ray_dim, 
                                                 proprio_dim= proprio_dim,
                                                 cfg=cfg
                                                )
    
    backbone_model = create_backbone_model(type=backbone_type,
                                           cfg=cfg)
    
    actor = SquashedGaussianPolicyHead(backbone_dim=cfg.model.backbone_hidden_sizes[-1],
                                actions_dim=cfg.env.act_dim,
                                hidden_sizes=cfg.model.policy_hidden_sizes)
    
    critic = DoubleQNet(backbone_dim=cfg.model.backbone_hidden_sizes[-1],
                        action_dim= cfg.env.act_dim,
                        hidden_sizes=cfg.model.value_hidden_sizes)
        
    return SACAgent(obs_embed_model=observation_model,
                    backbone_model=backbone_model,
                    actor=actor, critic=critic)
