package com.ecommerce.project.security;


import com.ecommerce.project.model.AppRole;
import com.ecommerce.project.model.Role;
import com.ecommerce.project.model.User;
import com.ecommerce.project.reprositories.RoleRepository;
import com.ecommerce.project.reprositories.UserRepository;
import com.ecommerce.project.security.jwt.AuthEntryPointJwt;
import com.ecommerce.project.security.jwt.AuthTokenFilter;
import java.util.Set;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpMethod;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.dao.DaoAuthenticationProvider;
import org.springframework.security.config.Customizer;
import org.springframework.security.config.annotation.authentication.configuration.AuthenticationConfiguration;
import org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityCustomizer;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.annotation.web.configurers.HeadersConfigurer.FrameOptionsConfig;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;

@Configuration
@EnableWebSecurity
//@EnableMethodSecurity
public class WebSecurityConfig {

  @Autowired
  UserDetailsService userDetailsService;
  @Autowired
  private AuthEntryPointJwt unauthorizedHandler;

  @Bean
  public AuthTokenFilter authenticationJwtTokenFilter() {
    return new AuthTokenFilter();
  }

  @Bean
  public DaoAuthenticationProvider authenticationProvider() {
    DaoAuthenticationProvider authProvider = new DaoAuthenticationProvider();
    authProvider.setUserDetailsService(userDetailsService);
    authProvider.setPasswordEncoder(passwordEncoder());
    return authProvider;
  }

  @Bean
  public PasswordEncoder passwordEncoder() {
    return new BCryptPasswordEncoder();
  }

  @Bean
  public AuthenticationManager authenticationManager(
      AuthenticationConfiguration authenticationConfiguration) throws Exception {
    return authenticationConfiguration.getAuthenticationManager();
  }

  @Bean
  public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
    http
        .cors(Customizer.withDefaults()) //enable mvc
        .csrf(AbstractHttpConfigurer::disable)
        .exceptionHandling(
            exception -> exception.authenticationEntryPoint(unauthorizedHandler))
        .sessionManagement(
            session ->
                session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
        .authorizeHttpRequests(authorizeRequests ->
        authorizeRequests.requestMatchers("/api/auth/**").permitAll()
            .requestMatchers("/v3/api-docs/**").permitAll()
            .requestMatchers("/h2-console/**").permitAll()
            .requestMatchers("/swagger-ui/**").permitAll()
            .requestMatchers("/api/public/**").permitAll()
            //.requestMatchers("/api/admin/**").permitAll()
            .requestMatchers("/api/test/**").permitAll()
            .requestMatchers("/images/**").permitAll()
            .anyRequest().authenticated());

    http.authenticationProvider(authenticationProvider());
    http.addFilterBefore(authenticationJwtTokenFilter(),
        UsernamePasswordAuthenticationFilter.class);
    http.headers(headers -> headers.frameOptions(
        FrameOptionsConfig::sameOrigin));
    return http.build();
  }

  @Bean
  public WebSecurityCustomizer webSecurityCustomizer() {
    return (web -> web.ignoring().requestMatchers(
        "/v2/api-dpcs",
        "/configuration/ui",
        "/configuration/security",
        "/webjars/**",
        "/swagger-resources/**",
        "/swagger-ui.html"));
  }

  @Bean
  public CommandLineRunner initData(RoleRepository roleRepository, UserRepository userRepository, PasswordEncoder passwordEncoder) {
    return args -> {
      // Retrieve or create roles
      Role userRole = roleRepository.findByRoleName(AppRole.ROLE_USER)
          .orElseGet(() -> {
            Role newUserRole = new Role(AppRole.ROLE_USER);
            return roleRepository.save(newUserRole);
          });

      Role sellerRole = roleRepository.findByRoleName(AppRole.ROLE_SELLER)
          .orElseGet(() -> {
            Role newSellerRole = new Role(AppRole.ROLE_SELLER);
            return roleRepository.save(newSellerRole);
          });

      Role adminRole = roleRepository.findByRoleName(AppRole.ROLE_ADMIN)
          .orElseGet(() -> {
            Role newAdminRole = new Role(AppRole.ROLE_ADMIN);
            return roleRepository.save(newAdminRole);
          });

      Set<Role> userRoles = Set.of(userRole);
      Set<Role> sellerRoles = Set.of(sellerRole);
      Set<Role> adminRoles = Set.of(userRole, sellerRole, adminRole);


      // Create users if not already present
      if (!userRepository.existsByUserName("user1")) {
        User user1 = new User("user1", "user1@example.com", passwordEncoder.encode("password1"));
        userRepository.save(user1);
      }

      if (!userRepository.existsByUserName("seller1")) {
        User seller1 = new User("seller1", "seller1@example.com", passwordEncoder.encode("password2"));
        userRepository.save(seller1);
      }

      if (!userRepository.existsByUserName("admin")) {
        User admin = new User("admin", "admin@example.com", passwordEncoder.encode("adminPass"));
        userRepository.save(admin);
      }

      // Update roles for existing users
      userRepository.findByUserName("user1").ifPresent(user -> {
        user.setRoles(userRoles);
        userRepository.save(user);
      });

      userRepository.findByUserName("seller1").ifPresent(seller -> {
        seller.setRoles(sellerRoles);
        userRepository.save(seller);
      });

      userRepository.findByUserName("admin").ifPresent(admin -> {
        admin.setRoles(adminRoles);
        userRepository.save(admin);
      });
    };
  }
}
