import logging

import torch

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, build_zero_shot_classifier_mad, IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES, IMAGENET_A_CLASSNAMES, IMAGENET_R_CLASSNAMES, CIFAR10_CLASSNAMES, CIFAR100_CLASSNAMES, FLOWERS_CLASSNAMES, STANFORD_CLASSNAMES, IMAGENET_O_CLASSNAMES, FOOD_CLASSNAMES

from open_clip_train.precision import get_autocast

import random
import mgap


def accuracy(output, target, topk=(1,)):
    
    max_k = output.shape[1]
    if max(topk) > max_k:
        topk = [min(k, max_k) for k in topk]
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args, use_recons=False, use_filtered=False, filter_algo=None, filter_obj=None, return_mgap_values = False):
    assert args.batch_size == args.train_mask_batch_size
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)
    text_features = classifier.t().cpu()

    
    text_features_norm = text_features / text_features.norm(dim=1, keepdim=True)
    

    ############## mgap metrics
    mgap_obj = mgap.metrics.ModalityGapLiang(embed_dim=args.embed_dim)
    # .to(args.device)
    rmg_obj = mgap.metrics.RMGSchrodi()

    
    all_part_recon_image_features = []
    all_text_features = []
    with torch.inference_mode():
        if filter_algo is not None:
            mask = filter_algo.get_mask(text_features.float())
        top1, top5, n = 0., 0., 0.
        count = 0 
        for batch_idx, (images, target) in enumerate(dataloader):

            images = images.to(device=device, dtype=input_dtype)

                # predict
            
            image_features = model.encode_image(images)
            
            if use_filtered and hasattr(args, "filter_arch_name") and args.filter_arch_name != "Identity":
                
                
                logits = torch.zeros((image_features.shape[0], text_features.shape[0]))
                for i in range(len(image_features)):
                    image_feature = image_features[i]

                    info = filter_obj.get_latents(image_feature)
                    latents, inds, vals = filter_obj.unpack_info(info)
                    

                    if inds is not None:                        
                        after_mask_vals = vals * mask[:, inds]

                        inds = inds.unsqueeze(0) # (128,) -> (1, 128) 
                        inds = inds.expand(mask.shape[0], inds.shape[1])# (1, 128,) -> (1000, 128) 
                        inds = inds.contiguous()

                   
                        info_updated = {"inds": inds, "vals": after_mask_vals,
                        "latents_pre_act": latents, "mask": mask}


                        part_recon_image_features = filter_obj.get_reconstructed_features(
                            image_feature, info_updated)
   
                    else:
                        part_recon_image_features = latents * mask

                    part_recon_image_features = part_recon_image_features / part_recon_image_features.norm(dim=-1, keepdim=True)
                    logits[i]= 100*(part_recon_image_features.detach().cpu() * text_features_norm).sum(dim=-1)


                    # For metric
                    part_recon_image_features_metric = part_recon_image_features[target[i]].float().detach().cpu().unsqueeze(dim=0)
                    text_features_norm_metric = text_features_norm[target[i]].float().detach().cpu().unsqueeze(dim=0) 
                    
                    
                    mgap_obj.update(part_recon_image_features_metric, text_features_norm_metric)


                    if count <2000: 
                        all_part_recon_image_features.append(part_recon_image_features_metric.half())
                        all_text_features.append(text_features_norm_metric.half())
                        count += part_recon_image_features_metric.shape[0]
                    

            elif use_recons and hasattr(args, "filter_arch_name") and args.filter_arch_name != "Identity":
                info = filter_obj.get_latents(image_features)
                recon_image_features = filter_obj.get_reconstructed_features(embeddings=image_features, info=info)
                logits = 100. * recon_image_features @ classifier_norm
            else:
                image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features_norm = text_features_norm.float()
                logits=  100*(image_features_norm.float().detach().cpu() @ text_features_norm.t())

                # For metric
                image_features_norm_metric = image_features_norm.float().detach().cpu()
                text_features_norm_metric = text_features_norm[target].float().detach().cpu()
                mgap_obj.update(image_features_norm_metric, text_features_norm_metric)

                if count <2000: 
                    all_part_recon_image_features.append(image_features_norm_metric.half())
                    all_text_features.append(text_features_norm_metric.half())
                    count += image_features_norm_metric.shape[0]
                

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)


    # 2000 number of samples to take, since batch size might not be a divisor of 2000
    all_part_recon_image_features = all_part_recon_image_features[0:2000]
    all_text_features = all_text_features[0:2000]

    all_part_recon_image_features = torch.cat(all_part_recon_image_features, dim=0)
    all_part_recon_image_features = all_part_recon_image_features.float()


    all_text_features = torch.cat(all_text_features, dim=0)
    all_text_fetaures = all_text_features.float()

    rmg_obj.update(all_part_recon_image_features, all_text_fetaures, normalized = False)


    all_mgap_rmg_vals= {}
    all_mgap_rmg_vals.update(mgap_obj.get_logging_dict(panel_name="val"))
    all_mgap_rmg_vals.update(rmg_obj.get_logging_dict(panel_name="val"))


    top1 = (top1 / n)
    top5 = (top5 / n)
    

    if return_mgap_values:
        return top1, top5, all_mgap_rmg_vals
    else: 
        return top1, top5


def zero_shot_eval(model, data, epoch, args, tokenizer=None, use_recons=False, use_filtered=False, filter_algo=None, filter_obj=None, normalize_embeddings=True, average_embeddings=True):
    if args.distributed and not args.horovod:
        model = model.module

    logging.info('Starting zero-shot imagenet.')
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    logging.info('Building zero-shot classifier')
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    with autocast():
        # IMAGENET_CLASSNAMES
        if 'imagenet-val' in data or 'imagenet-v2' in data or 'imagenet-sketch' in data:
            print("either imagenet-val or imagenet-v2 or imagenet-sketch found")
            imagenet_classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=IMAGENET_CLASSNAMES,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=10,
                device=device,
                use_tqdm=True,
                normalize_embeddings=normalize_embeddings,
                average_embeddings=average_embeddings,
            )

    logging.info('Using classifier')
    results = {}
    if 'imagenet-val' in data:
        top1, top5, all_mgap_rmg_vals = run(model, imagenet_classifier,
                         data['imagenet-val'].dataloader, args, use_recons=use_recons, use_filtered=use_filtered, filter_algo=filter_algo, filter_obj=filter_obj, return_mgap_values= True)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
        for key in all_mgap_rmg_vals:
            key_name = 'imagenet-val-' + key
            results[key_name] = all_mgap_rmg_vals[key]
    if 'imagenet-v2' in data:
        top1, top5, all_mgap_rmg_vals = run(model, imagenet_classifier,
                         data['imagenet-v2'].dataloader, args, use_recons=use_recons, use_filtered=use_filtered, filter_algo=filter_algo, filter_obj=filter_obj, return_mgap_values= True)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5
        for key in all_mgap_rmg_vals:
            key_name = 'imagenet-v2-' + key
            results[key_name] = all_mgap_rmg_vals[key]
    
    if 'imagenet-sketch' in data:
        top1, top5, all_mgap_rmg_vals = run(model, imagenet_classifier, data['imagenet-sketch'].dataloader, args, use_recons=use_recons, use_filtered=use_filtered, filter_algo=filter_algo, filter_obj=filter_obj, return_mgap_values= True)
        results['imagenet-sketch-zeroshot-val-top1'] = top1
        results['imagenet-sketch-zeroshot-val-top5'] = top5
        for key in all_mgap_rmg_vals:
            key_name = 'imagenet-sketch-' + key
            results[key_name] = all_mgap_rmg_vals[key]
    if 'imagenet-a' in data:
        classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=IMAGENET_A_CLASSNAMES,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=10,
                device=device,
                use_tqdm=True,
                normalize_embeddings=normalize_embeddings,
                average_embeddings=average_embeddings,
            )
        top1, top5, all_mgap_rmg_vals = run(model, classifier, data['imagenet-a'].dataloader, args, use_recons=use_recons, use_filtered=use_filtered, filter_algo=filter_algo, filter_obj=filter_obj, return_mgap_values= True)
        results['imagenet-a-zeroshot-val-top1'] = top1
        results['imagenet-a-zeroshot-val-top5'] = top5
        for key in all_mgap_rmg_vals:
            key_name = 'imagenet-a-' + key
            results[key_name] = all_mgap_rmg_vals[key]
    if 'imagenet-o' in data:
        classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=IMAGENET_O_CLASSNAMES,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=10,
                device=device,
                use_tqdm=True,
                normalize_embeddings=normalize_embeddings,
                average_embeddings=average_embeddings,
            )
        top1, top5, all_mgap_rmg_vals = run(model, classifier, data['imagenet-o'].dataloader, args, use_recons=use_recons, use_filtered=use_filtered, filter_algo=filter_algo, filter_obj=filter_obj, return_mgap_values= True)
        results['imagenet-o-zeroshot-val-top1'] = top1
        results['imagenet-o-zeroshot-val-top5'] = top5
        for key in all_mgap_rmg_vals:
            key_name = 'imagenet-o-' + key
            results[key_name] = all_mgap_rmg_vals[key]
    if 'imagenet-c' in data:
        raise ValueError('Unsupported dataset!')
        classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=TADA,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=10,
                device=device,
                use_tqdm=True,
                normalize_embeddings=normalize_embeddings,
                average_embeddings=average_embeddings,
            )
        top1, top5, all_mgap_rmg_vals = run(model, classifier, data['imagenet-c'].dataloader, args, use_recons=use_recons, use_filtered=use_filtered, filter_algo=filter_algo, filter_obj=filter_obj, return_mgap_values= True)
        results['imagenet-c-zeroshot-val-top1'] = top1
        results['imagenet-c-zeroshot-val-top5'] = top5
        for key in all_mgap_rmg_vals:
            key_name = 'imagenet-c-' + key
            results[key_name] = all_mgap_rmg_vals[key]
    if 'imagenet-r' in data:
        classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=IMAGENET_R_CLASSNAMES,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=10,
                device=device,
                use_tqdm=True,
                normalize_embeddings=normalize_embeddings,
                average_embeddings=average_embeddings,
            )
        top1, top5, all_mgap_rmg_vals = run(model, classifier, data['imagenet-r'].dataloader, args, use_recons=use_recons, use_filtered=use_filtered, filter_algo=filter_algo, filter_obj=filter_obj, return_mgap_values= True)
        results['imagenet-r-zeroshot-val-top1'] = top1
        results['imagenet-r-zeroshot-val-top5'] = top5
        for key in all_mgap_rmg_vals:
            key_name = 'imagenet-r-' + key
            results[key_name] = all_mgap_rmg_vals[key]

    if 'cifar10' in data:
        classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=CIFAR10_CLASSNAMES,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=10,
                device=device,
                use_tqdm=True,
                normalize_embeddings=normalize_embeddings,
                average_embeddings=average_embeddings,
            )
        top1, top5, all_mgap_rmg_vals = run(model, classifier, data['cifar10'].dataloader, args, use_recons=use_recons, use_filtered=use_filtered, filter_algo=filter_algo, filter_obj=filter_obj, return_mgap_values= True)
        results['cifar10-zeroshot-val-top1'] = top1
        results['cifar10-zeroshot-val-top5'] = top5
        for key in all_mgap_rmg_vals:
            key_name = 'cifar10-' + key
            results[key_name] = all_mgap_rmg_vals[key]
    if 'cifar100' in data:
        classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=CIFAR100_CLASSNAMES,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=10,
                device=device,
                use_tqdm=True,
                normalize_embeddings=normalize_embeddings,
                average_embeddings=average_embeddings,
            )
        top1, top5, all_mgap_rmg_vals = run(model, classifier, data['cifar100'].dataloader, args, use_recons=use_recons, use_filtered=use_filtered, filter_algo=filter_algo, filter_obj=filter_obj, return_mgap_values= True)
        results['cifar100-zeroshot-val-top1'] = top1
        results['cifar100-zeroshot-val-top5'] = top5
        for key in all_mgap_rmg_vals:
            key_name = 'cifar100-' + key
            results[key_name] = all_mgap_rmg_vals[key]
    if 'flowers-102' in data:
        classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=FLOWERS_CLASSNAMES,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=10,
                device=device,
                use_tqdm=True,
                normalize_embeddings=normalize_embeddings,
                average_embeddings=average_embeddings,
            )
        top1, top5, all_mgap_rmg_vals = run(model, classifier, data['flowers-102'].dataloader, args, use_recons=use_recons, use_filtered=use_filtered, filter_algo=filter_algo, filter_obj=filter_obj, return_mgap_values= True)
        results['flowers102-zeroshot-val-top1'] = top1
        results['flowers102-zeroshot-val-top5'] = top5
        for key in all_mgap_rmg_vals:
            key_name = 'flowers102-' + key
            results[key_name] = all_mgap_rmg_vals[key]
    if 'food-101' in data:
        classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=FOOD_CLASSNAMES,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=10,
                device=device,
                use_tqdm=True,
                normalize_embeddings=normalize_embeddings,
                average_embeddings=average_embeddings,
            )
        top1, top5, all_mgap_rmg_vals = run(model, classifier, data['food-101'].dataloader, args, use_recons=use_recons, use_filtered=use_filtered, filter_algo=filter_algo, filter_obj=filter_obj, return_mgap_values= True)
        results['food101-zeroshot-val-top1'] = top1
        results['food101-zeroshot-val-top5'] = top5
        for key in all_mgap_rmg_vals:
            key_name = 'food101-' + key
            results[key_name] = all_mgap_rmg_vals[key]
    if 'stanford' in data:
        classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=STANFORD_CLASSNAMES,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=10,
                device=device,
                use_tqdm=True,
                normalize_embeddings=normalize_embeddings,
                average_embeddings=average_embeddings,
            )
        top1, top5, all_mgap_rmg_vals = run(model, classifier, data['stanford'].dataloader, args, use_recons=use_recons, use_filtered=use_filtered, filter_algo=filter_algo, filter_obj=filter_obj, return_mgap_values= True)
        results['stanford-zeroshot-val-top1'] = top1
        results['stanford-zeroshot-val-top5'] = top5
        for key in all_mgap_rmg_vals:
            key_name = 'stanford-' + key
            results[key_name] = all_mgap_rmg_vals[key]
    

    logging.info('Finished zero-shot imagenet.')
    logging.info(f'The results are: {results}.')


    return results


def zero_shot_eval_mad_schrodi(model, data, epoch, args, classnames, tokenizer=None, attribute_name=None):

    logging.info(f'Starting zero-shot MAD. {attribute_name}')
    assert tokenizer is not None

    attribute_name_dict = {'thickthinning': ["no thickthinning", "thickening", "thinning"],
                           'swelling': ["no swelling", "swelling"],
                           'fracture': ["no fracture", "fracture"],
                           'scaling': ["large", "small"],
                           'colors': ["gray", "red", "green", "blue", "cyan", "magenta", "yellow"],
                           'digit': list(map(str, range(10)))
                           }
    remaining_attributes = [attr for attr in attribute_name_dict.keys() if attr not in [
        attribute_name, "digit"]]
    attr_count = int(args.mad_mode[-1])
    if attribute_name == "digit":
        selected_attributes = random.sample(remaining_attributes, attr_count)
    else:
        selected_attributes = ["digit"]
        if attr_count > 1:
            selected_attributes += random.sample(
                remaining_attributes, attr_count - 1)
    logging.info(
        f"Attribute name: {attribute_name}, Selected attributes: {selected_attributes}")
    attr_list = []
    for attr in selected_attributes:
        attr_options = attribute_name_dict[attr]
        sample_attr = random.choice(attr_options)
        attr_list.append(sample_attr)
    class_insert_index = random.randint(0, len(attr_list) - 1)
    classification_texts = []
    for classname in classnames:
        this_class_texts = attr_list[:class_insert_index] + \
            [classname] + attr_list[class_insert_index:]
        classification_texts.append("-".join(this_class_texts))
    logging.info(f"Classification texts: {classification_texts}")

    logging.info('Building zero-shot classifier')
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    with autocast():
        classifier = build_zero_shot_classifier_mad(
            model,
            tokenizer=tokenizer,
            classnames=classnames,
            device=device,
        )

    logging.info('Using classifier')
    results = {}
    data['mad-val'].dataloader.dataset.object = attribute_name
    top1, top5 = run(model, classifier, data['mad-val'].dataloader, args)
    results[f'{attribute_name}-mad-schrodi-zeroshot-val-top1'] = top1

    logging.info(f'Finished zero-shot mad. {attribute_name}')

    return results

