package com.ecommerce.project.controller;

import com.ecommerce.project.model.Cart;
import com.ecommerce.project.payload.CartDTO;
import com.ecommerce.project.payload.CartItemDTO;
import com.ecommerce.project.reprositories.CartRepository;
import com.ecommerce.project.service.CartService;
import com.ecommerce.project.util.AuthUtil;
import jakarta.transaction.Transactional;
import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;

import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class CartController {

  @Autowired
  private CartService cartService;
  @Autowired
  private AuthUtil authUtil;
  @Autowired
  private CartRepository cartRepository;

  @Transactional
  @PostMapping("/cart/create")
  public ResponseEntity<String> createOrUpdateCart(@RequestBody List<CartItemDTO> cartItemDTOS) {
    String response = cartService.createOrUpdateCartWithItems(cartItemDTOS);
    return new ResponseEntity<String>(response, HttpStatus.CREATED);
  }


  @PostMapping("/carts/products/{productId}/quantity/{quantity}")
  public ResponseEntity<CartDTO> addProductToCart(@PathVariable Long productId,
                                                  @PathVariable Integer quantity) {
    CartDTO cartDTO = cartService.addProductToChart(productId, quantity);
    return new ResponseEntity<CartDTO>(cartDTO, HttpStatus.CREATED);
  }

  @GetMapping("/carts")
  public ResponseEntity<List<CartDTO>> getAllCarts() {
    List<CartDTO> cartDTOS = cartService.getAllCarts();
    return new ResponseEntity<List<CartDTO>>(cartDTOS, HttpStatus.FOUND);
  }

  @GetMapping("/carts/users/cart")
  public ResponseEntity<CartDTO> getCartById() {
    String emailId = authUtil.loggedInEmail();
    Cart cart = cartRepository.findCartByEmail(emailId);
    Long cartId = cart.getCartId();
    CartDTO cartDTO = cartService.getCart(emailId, cartId);
    return new ResponseEntity<>(cartDTO, HttpStatus.OK);
  }

  @PutMapping("/carts/products/{productId}/quantity/{operation}")
  public ResponseEntity<CartDTO> updateCartProduct(@PathVariable Long productId,
                                                    @PathVariable String operation) {
    CartDTO cartDTO = cartService.updateProductQuantityInCart(productId,
        operation.equalsIgnoreCase("delete") ? -1 : 1);
    return new ResponseEntity<>(cartDTO, HttpStatus.OK);
  }

  @DeleteMapping("/carts/{cartId}/product/{productId}")
  public ResponseEntity<String> deleteProductFromCart(@PathVariable Long cartId,
                                                    @PathVariable Long productId) {
    String status = cartService.deleteProductFromCart(cartId, productId);
    return new ResponseEntity<>(status, HttpStatus.OK);
  }


}
