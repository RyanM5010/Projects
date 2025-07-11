package com.ecommerce.project.service;

import com.ecommerce.project.payload.StripePaymentDTO;
import com.stripe.Stripe;
import com.stripe.exception.StripeException;
import com.stripe.model.PaymentIntent;
import com.stripe.param.PaymentIntentCreateParams;
import jakarta.annotation.PostConstruct;
import jakarta.transaction.Transactional;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

@Transactional
@Service
public class StripeServiceImpl implements StripeService {

  @Value("${stripe.secret.key}")
  private String stripeKey;

  @PostConstruct
  public void init() {
    Stripe.apiKey = stripeKey;
  }

  @Override
  public PaymentIntent paymentIntent(StripePaymentDTO stripePaymentDto) throws StripeException {
    PaymentIntentCreateParams params =
        PaymentIntentCreateParams.builder().setAmount(stripePaymentDto.getAmount())
            .setCurrency(stripePaymentDto.getCurrency()).build();
    return PaymentIntent.create(params);
  }
}
