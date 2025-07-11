package com.ecommerce.project.service;

import com.ecommerce.project.exceptions.APIException;
import com.ecommerce.project.exceptions.ResourceNotFoundException;
import com.ecommerce.project.model.Cart;
import com.ecommerce.project.model.CartItem;
import com.ecommerce.project.model.Product;
import com.ecommerce.project.payload.CartDTO;
import com.ecommerce.project.payload.CartItemDTO;
import com.ecommerce.project.payload.ProductDTO;
import com.ecommerce.project.reprositories.CartItemRepository;
import com.ecommerce.project.reprositories.CartRepository;
import com.ecommerce.project.reprositories.ProductRepository;
import com.ecommerce.project.util.AuthUtil;
import jakarta.transaction.Transactional;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.modelmapper.ModelMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class CartServiceImpl implements CartService {
  @Autowired
  CartRepository cartRepository;

  @Autowired
  ProductRepository productRepository;

  @Autowired
  CartItemRepository cartItemRepository;

  @Autowired
  ModelMapper modelMapper = new ModelMapper();

  @Autowired
  AuthUtil authUtil;

  @Override
  public CartDTO addProductToChart(Long productId, Integer quantity) {
    Cart cart  = createCart();

    Product product = productRepository.findById(productId)
        .orElseThrow(() -> new ResourceNotFoundException("Product", "productId", productId));

    CartItem cartItem = cartItemRepository.findCartItemByCartIdAndProductId(cart.getCartId(), productId);

    if (cartItem != null) {
      throw new APIException("Product " + product.getProductName() + " already exists in the cart");
    }

    if (product.getQuantity() == 0) {
      throw new APIException(product.getProductName() + " is not available");
    }

    if (product.getQuantity() < quantity) {
      throw new APIException("Please, make an order of the " + product.getProductName()
          + " less than or equal to the quantity " + product.getQuantity() + ".");
    }

    CartItem newCartItem = new CartItem();

    newCartItem.setProduct(product);
    newCartItem.setCart(cart);
    newCartItem.setQuantity(quantity);
    newCartItem.setDiscount(product.getDiscount());
    newCartItem.setProductPrice(product.getSpecialPrice());

    cartItemRepository.save(newCartItem);

    product.setQuantity(product.getQuantity());

    cart.setTotalPrice(cart.getTotalPrice() + (product.getSpecialPrice() * quantity));

    cartRepository.save(cart);

    CartDTO cartDTO = modelMapper.map(cart, CartDTO.class);

    List<CartItem> cartItems = cart.getCartItems();

    Stream<ProductDTO> productStream = cartItems.stream().map(item -> {
      ProductDTO map = modelMapper.map(item.getProduct(), ProductDTO.class);
      map.setQuantity(item.getQuantity());
      return map;
    });

    cartDTO.setProducts(productStream.toList());

    return cartDTO;
  }


  @Override
  public List<CartDTO> getAllCarts() {
    List<Cart> carts = cartRepository.findAll();
    if (carts.size()== 0) {
      throw new APIException("No carts exist");
    }
    List<CartDTO> cartDTOS = carts.stream().map(
        cart -> {
          CartDTO cartDTO = modelMapper.map(cart, CartDTO.class);
          List<ProductDTO> products = cart.getCartItems().stream().map(cartItem -> {
            ProductDTO productDTO = modelMapper.map(cartItem.getProduct(), ProductDTO.class);
            productDTO.setQuantity(cartItem.getQuantity());
            return productDTO;
          }).toList();
          cartDTO.setProducts(products);
          return cartDTO;
        }).toList();
    return cartDTOS;
  }

  @Override
  public CartDTO getCart(String emailId, Long cartId) {
    Cart cart = cartRepository.findCartByEmailAndCartId(emailId, cartId);
    if (cart == null){
      throw new ResourceNotFoundException("Cart", "cartId", cartId);
    }
    CartDTO cartDTO = modelMapper.map(cart, CartDTO.class);
    cart.getCartItems().forEach(c ->
        c.getProduct().setQuantity(c.getQuantity()));
    List<ProductDTO> products = cart.getCartItems().stream()
        .map(p -> modelMapper.map(p.getProduct(), ProductDTO.class))
        .toList();
    cartDTO.setProducts(products);
    return cartDTO;
  }

  @Transactional
  @Override
  public CartDTO updateProductQuantityInCart(Long productId, Integer quantity) {
    String emailId= authUtil.loggedInEmail();
    Cart userCart = cartRepository.findCartByEmail(emailId);
    Long userCartId = userCart.getCartId();

    Cart cart = cartRepository.findById(userCartId).orElseThrow(
        () -> new ResourceNotFoundException("Cart", "cartId",userCartId));

    Product product = productRepository.findById(productId)
        .orElseThrow( () ->
            new ResourceNotFoundException("Product", "productId", productId));
    if (product.getQuantity() == 0)
      throw new APIException(product.getProductName() + " is not available in stock");
    if (product.getQuantity() < quantity) {
      throw new APIException("Please make an order of the " + product.getProductName()
          + " less than or equal to the quantity" + product.getQuantity());
    }
    CartItem cartItem = cartItemRepository.findCartItemByCartIdAndProductId(userCartId, productId);
    if (cartItem == null) {
      throw new APIException("Product " + product.getProductName() + " not available in the cart");
    }
    int newQuantity = cartItem.getQuantity() + quantity;
    if (newQuantity < 0){
      throw new APIException("The resulting quantity is less than or equal to zero");
    }
    if (newQuantity == 0){
      deleteProductFromCart(userCartId, productId);
    } else {
      cartItem.setProductPrice(product.getSpecialPrice());
      cartItem.setQuantity(cartItem.getQuantity() + quantity);
      cartItem.setDiscount(product.getDiscount());
      cart.setTotalPrice(cart.getTotalPrice() + (cartItem.getProductPrice() * quantity));
      cartRepository.save(cart);
    }
    CartItem updatedCartItem = cartItemRepository.save(cartItem);
    if (updatedCartItem.getQuantity() == 0) {
      cartItemRepository.delete(updatedCartItem);
    }
    CartDTO cartDTO = modelMapper.map(cart, CartDTO.class);
    List<CartItem> cartItems = cart.getCartItems();
    Stream<ProductDTO> productDTOStream = cartItems.stream().map(
        item -> {
          ProductDTO map = modelMapper.map(item.getProduct(), ProductDTO.class);
          map.setQuantity(item.getQuantity());
          return map;
        });
    cartDTO.setProducts(productDTOStream.toList());
    return cartDTO;
  }

  @Transactional
  @Override
  public String deleteProductFromCart(Long cartId, Long productId) {
    Cart cart = cartRepository.findById(cartId).orElseThrow(
        () -> new ResourceNotFoundException("Cart", "cartId", cartId)
    );
    CartItem cartItem = cartItemRepository.findCartItemByCartIdAndProductId(cartId, productId);
    if (cartItem == null) {
      throw new ResourceNotFoundException("Product", "productId", productId);
    }
    cart.setTotalPrice(cart.getTotalPrice() - (cartItem.getProductPrice() * cartItem.getQuantity()));
    cartItemRepository.deleteCartItemByCartIdAndProductId(cartId, productId);
    return "Product " + cartItem.getProduct().getProductName() + " has been deleted";
  }

  @Override
  public void updateProductInCarts(Long cartId, Long productId) {
    Cart cart = cartRepository.findById(cartId).orElseThrow(
        () -> new ResourceNotFoundException("Cart", "cartId",cartId));
    Product product = productRepository.findById(productId)
        .orElseThrow( () ->
            new ResourceNotFoundException("Product", "productId", productId));

    CartItem cartItem = cartItemRepository.findCartItemByCartIdAndProductId(cartId, productId);
    if (cartItem == null) {
      throw new APIException("Product " + product.getProductName() + " not available in the cart");
    }
    double cartPrice = cart.getTotalPrice()
        - (cartItem.getProductPrice() * cartItem.getQuantity());
    cartItem.setProductPrice(product.getSpecialPrice());
    cart.setTotalPrice(cartPrice + (cartItem.getProductPrice() * cartItem.getQuantity()));
    cartItem = cartItemRepository.save(cartItem);
  }

  @Override
  public String createOrUpdateCartWithItems(List<CartItemDTO> cartItemDTOS) {
    String emailId = authUtil.loggedInEmail();
    Cart existingCart = cartRepository.findCartByEmail(emailId);
    if (existingCart == null) {
      existingCart = new Cart();
      existingCart.setTotalPrice(0.0);
      existingCart.setUser(authUtil.loggedInUser());
      existingCart = cartRepository.save(existingCart);
    } else {
      cartItemRepository.deleteAllByCartId(existingCart.getCartId());
    }
    double totalPrice = 0.00;
    for (CartItemDTO cartItemDTO : cartItemDTOS) {
      Long productId = cartItemDTO.getProductId();
      Integer quantity = cartItemDTO.getQuantity();

      Product product = productRepository.findById(productId)
          .orElseThrow(() -> new ResourceNotFoundException("Product", "productId", productId));
//      product.setQuantity(product.getQuantity() - quantity); after user checkout
      totalPrice += product.getSpecialPrice() * quantity;

      CartItem cartItem = new CartItem();
      cartItem.setProduct(product);
      cartItem.setQuantity(quantity);
      cartItem.setCart(existingCart);
      cartItem.setProductPrice(product.getSpecialPrice());
      cartItem.setDiscount(product.getDiscount());
      cartItemRepository.save(cartItem);
    }
    existingCart.setTotalPrice(totalPrice);
    cartRepository.save(existingCart);
    return "Cart created/updated with new items";
  }

  private Cart createCart() {
    Cart userCart = cartRepository.findCartByEmail((authUtil.loggedInEmail()));
    if (userCart != null)
      return userCart;
    Cart cart = new Cart();
    cart.setTotalPrice(0.00);
    cart.setUser(authUtil.loggedInUser());
    Cart newCart = cartRepository.save(cart);
    return newCart;
  }


}