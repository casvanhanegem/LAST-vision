"""Predict with an explicit readout; accepts two-stage v2 and v3 checkpoints."""
import argparse
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from visual_snn.encoders import make_encoder
from visual_snn.model import make_model
from visual_snn.readout import scores


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--readout', choices=['membrane', 'spike', 'paper'],
                   help='Explicit override; by default use checkpoint training protocol')
    args = p.parse_args()
    saved = torch.load(args.checkpoint, weights_only=True, map_location=args.device)
    if saved.get('implementation') not in ['last-vision-v2', 'last-vision-v3']:
        raise SystemExit('This command requires a two-stage v2 or v3 checkpoint.')
    c = saved['config']
    encoder = make_encoder(c).to(args.device).eval()
    model = make_model(c).to(args.device).eval()
    encoder.load_state_dict(saved['encoder'])
    model.load_state_dict(saved['model'])
    mode, size = ('L', 28) if c['dataset'] == 'mnist' else ('RGB', 32)
    image = to_tensor(Image.open(args.image).convert(mode).resize((size, size)))[None].to(args.device)
    with torch.inference_mode():
        generator = torch.Generator(device=args.device).manual_seed(c['eval_seed'])
        spikes, membranes, _ = model(encoder(image, generator=generator))
    readout = args.readout or c.get('readout', 'paper')
    print('Readout:', readout)
    print('Predicted class index:', scores(spikes, membranes, readout).argmax(1).item())
    print('Output spikes:', spikes.sum().item())
