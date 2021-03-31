import re

stop_words = ['bạn', 'ban', 'anh', 'chị', 'chi', 'em', 'shop', 'bot', 'ad']

def convert_to_no_accents(text):
    patterns = {
        '[àáảãạăắằẵặẳâầấậẫẩ]': 'a',
        '[đ]': 'd',
        '[èéẻẽẹêềếểễệ]': 'e',
        '[ìíỉĩị]': 'i',
        '[òóỏõọôồốổỗộơờớởỡợ]': 'o',
        '[ùúủũụưừứửữự]': 'u',
        '[ỳýỷỹỵ]': 'y'
    }
    output = text
    for regex, replace in patterns.items():
        output = re.sub(regex, replace, output)
        output = re.sub(regex.upper(), replace.upper(), output)
    return output

if __name__ == '__main__':
    a = '''
    - Trẻ em dưới 6 tuổi chưa được cấp thẻ bảo hiểm y tế thì cần xuất trình giấy tờ gì khi đi khám, chữa bệnh?
    - Doanh nghiệp có từ bao nhiêu lao động trở lên mới phải đóng bảo hiểm xã hội?
    - Tội trốn đóng bảo hiểm xã hội, bảo hiểm y tế, bảo hiểm thất nghiệp thì bị xử lý như thế nào?
    - Tôi có thời gian công tác trước năm 1995 chưa được ghi trên sổ bảo hiểm xã hội, tôi cần hồ sơ thủ tục như thế nào để được cộng nối?
    - Tôi muốn hỏi để cấp lại sổ bảo hiểm xã hội thành phần hồ sơ gồm những gì?
    - Cách tính mức trợ cấp tai nạn lao động đối với người lao động đã hưởng trợ cấp tai nạn lao động một lần từ ngày 01 tháng 01 năm 2007 được giám định lại mức suy giảm khả năng lao động sau khi thương tật tái phát được quy định như thế nào?
    - Xin hỏi lao động nữ năm nay 45 tuổi, có 21 năm đóng bảo hiểm xã hội trong đó có 16 năm làm công việc đặc biệt nặng nhọc độc hại nguy hiểm, bị suy giảm khả năng lao động 61%. Tôi muốn nghỉ việc để hưởng chế độ hưu trí có được không?
    - Tiền lương tháng đóng bảo hiểm thất nghiệp đối với người lao động thuộc đối tượng thực hiện chế độ tiền lương do Nhà nước quy định tối đa là bao nhiêu?
    - Tôi muốn biết người tham gia bảo hiểm y tế thanh toán trực tiếp chi phí khám, chữa bệnh tại cơ quan bảo hiểm xã hội trong trường hợp nào?
    - Công ty tôi chậm nộp bảo hiểm xã hội thì có bị tính lãi không?
    - Hiện nay, xin cấp Giấy phép xuất nhập khẩu từ nội địa vào Khu vực hải quan riêng có cần xin cấp Giấy phép không?
    - Hồ sơ thông báo tập trung kinh tế gồm những giấy tờ gì?
    - Đề nghị cho biết Hồ sơ đề nghị cấp mới Giấy chứng nhận đăng ký hoạt động giám định?
    - Đề nghị cho biết Hồ sơ đề nghị cấp mới Giấy chứng nhận đăng ký hoạt động chứng nhận
    - Hồ sơ đề xuất bổ sung đề án  thực hiện Chương trình cấp quốc gia về xúc tiến thương mại gồm những hồ sơ gì? Có khác với hồ sơ đề xuất ban đầu không?
    - Tôi muốn hỏi về quy trình, thủ tục cấp lại C/O như thế nào
    - Thời hạn cấp sửa đổi, bổ sung/ cấp lại Giấy chứng nhận lưu hành tự do (CFS) đối với hàng hóa xuất khẩu là bao lâu?
    - Tôi là một doanh nghiệp nằm trên địa bàn tỉnh Long An. Theo kế hoạch kinh doanh, doanh nghiệp của tôi rất muốn mở rộng sang lĩnh vực xuất khẩu gạo. Tôi đề nghị Bộ Công Thương giúp hướng dẫn các quy định, cơ chế và chính sách liên quan để doanh nghiệp của tôi có thể xuất khẩu gạo ra nước ngoài trong thời gian tới?
    - Hình thức kiểm tra và đánh giá kết quả kiểm tra về kiến thức pháp luật bán hàng đa cấp được thực hiện như thế nào?
    - Những trường hợp nào phải sửa đổi, bổ sung giấy phép
    - Tôi muốn biết thủ tục cho phép trường trung học phổ thông chuyên hoạt động trở lại, có cách thức thực hiện như thế nào?
    - Thời hạn giải quyết việc mở ngành đào tạo trình độ tiến sĩ như thế nào?
    - Tôi muốn biết thủ tục cho phép trường trung học phổ thông chuyên hoạt động giáo dục, theo căn cứ pháp lý nào?
    - Tôi muốn biết thủ tục cho phép trường phổ thông dân tộc nội trú có cấp học cao nhất là trung học phổ thông hoạt động giáo dục, đối tượng nào thực hiện thủ tục hành chính?
    - Thủ tục cho phép trường trung học phổ thông hoạt động giáo dục, có trình tự thực hiện như thế nào?
    - Thuyên chuyển trường cho học sinh tiểu học, thực hiện là cơ quan nào?
    - Tôi muốn biết thủ tục cho phép trường phổ thông dân tộc nội trú có cấp học cao nhất là trung học phổ thông hoạt động giáo dục, cách thức thực hiện như thế nào?
    - Có mất phí và lệ phí đối với thủ tục hỗ trợ học tập đối với học sinh tiểu học các dân tộc thiểu số rất ít người?
    - Tôi muốn biết thủ tục cho phép trường trung học phổ thông chuyên hoạt động trở lại, có tên mẫu đơn và mẫu tờ khai không?
    - Tôi muốn biết thủ tục cho phép trường trung học phổ thông chuyên hoạt động trở lại, có thành phần, số lượng bộ hồ sơ như thế nào?
    - Trường hợp nào được cấp lại Giấy chứng nhận đủ điều kiện kinh doanh khai thác cảng biển?
    - Trình tự và thời gian cho ý kiến xây dựng công trình bảo đảm an ninh, quốc phòng trên đường thuỷ nội địa quốc gia?
    - Tổ hợp xe đầu kéo 3 trục kéo rơmooc 4 trục chở được bao nhiêu tấn hàng?
    - Tôi có thể nộp đơn để xin cấp Giấy chứng nhận thành viên tổ bay hay không? Nếu có thủ tục như thế nào?
    - Tôi có thể đến đâu để làm thủ tục xin cấp lại giấy phép hoạt động bến thủy nội địa?
    - Trường hợp nào được đổi GPLX quân sự sang GPLX ngành GTVT
    - Tôi muốn xóa đăng ký văn bản IDERA, hồ sơ đăng ký như thế nào?
    - Tôi cần chuẩn bị giấy tờ gì để thực hiện thủ tục công bố hoạt động cảng thủy nội địa?
    - Thời gian thực hiện thủ tục đăng ký cho tàu khách cao tốc vào hoạt động cố định trên tuyến là bao lâu?
    - Trường hợp nào là trường hợp hạn chế giao thông đường thuỷ nội địa?
    - Mẫu báo cáo khoản viện trợ phi dự án viện trợ như thế nào?
    - Đề nghị hướng dẫn về người có thẩm quyền xác thực hồ sơ đăng ký doanh nghiệp qua mạng điện tử?
    - Hồ sơ đăng ký doanh nghiệp có bắt buộc phải đóng dấu không?
    - Trường hợp công ty tại nước ngoài đã có giấy phép thành lập/ dự án có văn bản chấp thuận đầu tư tại nước ngoài trước ngày cấp Giấy chứng nhận đăng ký đầu tư ra nước ngoài thì Thông báo hoạt động đầu tư ở nước ngoài nộp khi nào?
    - Đầu tư ra nước ngoài là gì?
    - Những nội dung nào cần làm rõ khi lập Văn kiện dự án hỗ trợ kỹ thuật, phi dự án?
    - Khi có thay đổi về tên, địa chỉ trụ sở chính, ngành, nghề kinh doanh, vốn điều lệ, người đại diện theo pháp luật thì liên hiệp hợp tác xã phải tiến hành thủ tục thay đổi gì với cơ quan quản lý liên hiệp hợp tác xã kể từ ngày Thông tư số 07/2019/TT-BKHĐT có hiệu lực thi hành ngày 28/5/2019?
    - Nhà đầu tư không nộp hoặc nộp báo cáo chậm có bị xử phạt không?
    - Thanh toán không thành công (với thẻ nội địa hoặc thẻ quốc tế) thì hướng giải quyết tiếp theo như thế nào?
    - TTHC chuyển cơ sở bảo trợ xã hội, quỹ xã hội, quỹ từ thiện thành Doanh nghiệp xã hội
    - Trường hợp cơ sở đào tạo là tổ chức chứng nhận, đã xây dựng, áp dụng hệ thống quản lý theo tiêu chuẩn ISO/IEC 17021 thì bằng chứng cho việc áp dụng hệ thống quản lý theo tiêu chuẩn ISO 9001 đối với lĩnh vực đào tạo như thế nào?
    - Các tổ chức chứng nhận tại nước ngoài (không có trụ sở tại Việt Nam) sẽ phải thực hiện như thế nào để có thể đăng ký hoạt động chứng nhận theo quy định tại Nghị định số 107/2016/NĐ-CP? Kết quả chứng nhận được thừa nhận, chấp nhận tại Việt Nam?
    - Tôi là người nộp đơn Việt Nam thì tổ chức, cá nhân nào có thể đại diện cho tôi nộp đơn đăng ký xác lập quyền SHCN?
    - Doanh nghiệp đã được cấp Giấy chứng nhận quyền sử dụng Mã số, mã vạch khi sử dụng mã đã được cấp đó cấp mã cho sản phẩm của mình và in lên bao bì sản phẩm, lưu hành trên thị trường mà không khai báo và cập nhật các mã thương phẩm toàn cầu với có quan chức năng có bị xử phạt không?
    - Thủ tục cấp giấy phép tiến hành công việc bức xạ (sử dụng chất phóng xạ). Phí, lệ phí bao gồm?
    - Tôi cần lưu ý gì khi nộp bản đồ khu vực địa lý tương ứng với chỉ dẫn địa lý?
    - Khi làm hồ sơ Cấp đổi Giấy chứng nhận quyền sử dụng Mã số, mã vạch doanh nghiệp có cần nộp lại giấy chứng nhận bản gốc không?
    - Tôi muốn chuyển nhượng đơn đăng ký nhãn hiệu thì tôi cần nộp những tài liệu gì?
    - Tôi muốn biết các thiếu sót thường gặp đối với bản mô tả sáng chế nói chung?
    - Để nộp hồ sơ ghi nhận tổ chức dịch vụ đại diện sở hữu công nghiệp, tôi cần nộp những tài liệu nào ?
    - Khi doanh nghiệp đăng ký nội quy lao động thì cơ quan quản lý nhà nước có phải trả lời bằng văn bản cho doanh nghiệp không?
    - Thủ tục hỗ trợ làm nhà ở, sửa chữa nhà ở như thế nào?
    - Trình tự thực hiện thủ tục đăng ký nội quy lao động gồm những bước nào?
    - Công ty tôi được Sở Kế hoạch và Đầu tư Hà Nội cấp giấy chứng nhận đăng ký kinh doanh. Nay tôi muốn nộp hồ sơ xin cấp giấy chứng nhận đủ điều kiện  huấn luyện an toàn, vệ sinh lao động hạng C thì tôi sẽ phải nộp hồ sơ cho cơ quan nào để thực hiện việc cấp giấy chứng nhận huấn luyện bổ sung phạm vi huấn luyện?
    - Thời hạn giải quyết hỗ trợ chi phí mai táng là bao lâu?
    - Thủ tục tiếp nhận đối tượng bảo trợ có hoàn cảnh đặc biệt khó khăn vào cơ sở trợ giúp xã hội cấp tỉnh, gồm những nội dung gì?
    - Khi doanh nghiệp thay đổi người đại diện theo pháp luật của doanh nghiệp và cần được cấp lại giấy phép hoạt động cho thuê lại lao động thì phải nộp các giấy tờ gì cho Sở Lao động – Thương binh và Xã hội?
    - Những người như thế nào thì được hưởng chính sách trợ giúp xã hội thường xuyên tại cộng đồng?
    - Tôi có thể nộp hồ sơ xin cấp giấy  chứng nhận đủ điều kiện huấn luyện qua đường bưu điện được không?
    - Thời hạn giải quyết hồ sơ đề nghị cấp, cấp lại Chứng chỉ kiểm định viên là bao lâu? Quy định tại văn bản nào?
    - Tôi có thể xin cấp thị thực dài hạn với mục đích kinh doanh, thương mại tại Đại sứ quán được không ?
    - Tôi có thể xin chứng thực bản sao đơn ly hôn của chồng tôi được không?
    - Người nước ngoài có được cấp thị thực tại cửa khẩu quốc tế của Việt Nam hay không?
    - Chúng tôi sinh con ngoài giá thú. Tôi muốn ghi phần khai về cha trong Giấy khai sinh của trẻ thì cần làm thủ tục gì?
    - Con tôi mới sinh được hai tháng, nay muốn làm hộ chiếu để cháu đi du lịch với gia đình được không?
    - Tôi có thể ủy quyền cho người khác làm các thủ tục liên quan đến bất động sản ở Việt Nam?
    - Hồ sơ xin xác nhận nguồn gốc Việt Nam gồm những gì?
    - Sau khi đăng ký, công dân có nhận được giấy tờ chứng minh gì không?
    - Trường hợp bị mất hộ chiếu Việt Nam, Giấy xác nhận đăng ký công dân có giá trị như thế nào?
    - Người nhà tôi vi phạm pháp luật, đang thụ án tại nước ngoài. Khi hết hạn tù thì hộ chiếu đã quá hạn sử dụng. Người nhà tôi về nước bằng giấy tờ gì?
    - Điều kiện và tiêu chuẩn của người được đăng ký dự tuyển vào công chức là gì?
    - Đối với các hoạt động tôn giáo không có trong danh mục đã thông báo thì có cần thông báo bổ sung không? Việc thông báo bổ sung danh mục hoạt động tôn giáo được thực hiện như thế nào?
    - Đề nghị cho biết trình tự thực hiện thủ tục tặng thưởng Bằng khen cấp Bộ, ban, ngành, đoàn thể Trung ương, tỉnh, thành phố trực thuộc Trung ương về thành tích đối ngoại
    - Hồ sơ đầy đủ, hợp pháp đề nghị công nhận quỹ đủ điều kiện hoạt động và công nhận thành viên Hội đồng quản lý quỹ được giải quyết trong bao nhiêu ngày?
    - Đề nghị cho biết thời hạn thực hiện thủ tục tặng thưởng Bằng khen cấp Bộ, ban, ngành, đoàn thể Trung ương, tỉnh, thành phố trực thuộc Trung ương  cho gia đình
    - Đề nghị cho biết trình tự thực hiện thủ tục Khen thưởng "Huân chương Hồ Chí Minh" cho cá nhân có quá trình cống hiến
    - Đề nghị cho biết thành phần, số lượng hồ sơ khi thực hiện thủ tục Khen thưởng "Huân chương Hữu Nghị" cho tổ chức nước ngoài, cá nhân người nước ngoài
    - Khi xét chuyển cán bộ, công chức cấp xã thành công chức cấp huyện trở lên thì có phải thành lập Hội đồng kiểm tra, sát hạch không? Hội đồng kiểm tra, sát hạch có bao nhiêu thành viên. Nhiệm vụ, quyền hạn của Hội đồng kiểm tra, sát hạch là gì?
    - Tập thể, cá nhân phải đạt tiêu chuẩn như nào để được đề nghị xem xét tặng thưởng “Huân chương Độc lập” hạng ba?
    - Quỹ giải thể trong trường hợp nào?
    - Hồ sơ thực hiện thủ tục Xác nhận nguyên liệu thủy sản khai thác trong nước ?
    - Đơn vị nào được giao dự toán và ký hợp đồng thực hiện nhiệm vụ khuyến nông thường xuyên?
    - Thành phần hồ sơ đăng ký kiểm dịch động vật, sản phẩm động vật thủy sản tạm nhập tái xuất, tạm xuất tái nhập, chuyển cửa khẩu, kho ngoại quan, quá cảnh lãnh thổ Việt Nam quy định như thế nào?
    - Thời hạn xử lý xác nhận cam kết hoặc chứng nhận sản phẩm thủy sản xuất khẩu có nguồn gốc từ thủy sản khai thác nhập khẩu là bao lâu?
    - Thành phần hồ sơ đề nghị cấp văn bản chấp thuận cho tàu cá khai thác thuỷ sản ở vùng biển ngoài vùng biển Việt Nam hoặc cấp phép cho đi khai thác tại vùng biển thuộc thẩm quyền quản lý của Tổ chức nghề cá khu vực?
    - Đề nghị quý cơ quan cho biết, Ủy ban nhân dân cấp huyện có được điều chỉnh thiết kế, dự toán công trình lâm sinh không?
    - Trình tự Cấp giấy phép nhập khẩu tàu cá?
    - Đề nghị cho biết cơ quan và phương pháp xác định động vật rừng, thực vật rừng nguy cấp, quy, hiếm có nguồn gốc từ trại nuôi và từ tự nhiên.
    - Trình tự thực hiện thẩm tra thiết BVTC, dự toán xây dựng như thế nào?
    - Đề nghị cho biết hồ sơ đăng ký khảo nghiệm thuốc thú y gồm những tài liệu gì?
    - Cách tính tuổi nghề cơ yếu như thế nào?
    - Công dân Việt Nam xuất, nhập cảnh qua cửa khẩu biên giới đất liền phải có một trong những loại giấy tờ gì?
    - Trình tự thực hiện thủ tục hỗ trợ về nhà ở đối với thân nhân liệt sĩ đang công tác trong Quân đội có khó khăn về nhà ở theo Quyết định 4696/QĐ-BQP ngày 05/11/2015 của Bộ trưởng Bộ Quốc phòng?
    - Trình tự, thủ tục tổ chức hội nghị, hội thảo quốc tế trong Quân đội như thế nào?
    - Yêu cầu và điều kiện thực hiện thủ tục giải quyết chế độ trợ cấp một lần đối với dân quân tập trung ở miền Bắc, du kích tập trung ở miền Nam (bao gồm cả lực lượng mật quốc phòng)?
    - Tôi vừa là vợ của liệt sĩ (nay đã tái giá), vừa là mẹ của liệt sĩ. Tỏi đang hưởng trợ cấp tuất hàng tháng của liệt sĩ là con trai tôi. Nay tôi cỏ được hưởng thêm chế độ trợ cấp tuất đối với vợ liệt sĩ đã tải giả hay không?
    - Lực lượng dự bị động được hiểu như thế nào?
    - Thủ tục đăng ký xe điều động giữa các cơ quan, đơn vị đầu mối trực thuộc Bộ Quốc phòng?
    - Người nước ngoài nhập, xuất cảnh qua cửa khẩu biên giới đất liền Việt Nam phải có các loại giấy tờ gì?
    - Trách nhiệm của Cục BVAN QĐ đối với việc tổ chức hội nghị, hội thảo quốc tế trong Quân đội?
    - Giấy chứng nhận đăng ký hành nghề kiểm toán của tôi sắp hết hạn, tôi cần làm thủ tục gì để được gia hạn?
    - Tôi muốn hỏi nơi nộp hồ sơ khai thuế đối cá nhân có tài sản cho thuê là ở đâu?
    - Hồ sơ chào bán cổ phiếu riêng lẻ bao gồm những gì?
    - Tôi thực hiện thủ tục “Đề nghị áp dụng APA chính thức” bằng những cách nào?
    - Người nộp thuế nộp hồ sơ khai thuế Giá trị gia tăng đối với cơ sở sản xuất thủy điện hạch toán phụ thuộc EVN, trường hợp nhà máy thủy điện nằm trên 1 tỉnh ở đâu?
    - Công ty muốn thực hiện điều chỉnh Giấy chứng nhận đủ điều kiện kinh doanh xổ số cần chuẩn bị hồ sơ như thế nào?
    - Tôi muốn xin miễn thuế nhập khẩu đối với hàng hóa nhập khẩu phục vụ nghiên cứu khoa học, phát triển công nghệ. Tôi cần chuẩn bị hồ sơ gồm những gì?
    - Người nộp thuế thực hiện thủ tục “Rút đơn và dừng đàm phán APA”, gửi bao nhiêu bộ hồ sơ?
    - Tôi muốn đăng ký sửa đổi, bổ sung Danh mục hàng hóa nhập khẩu dung môi N-Hexan dùng trong sản xuất khô dầu đậu tương và dầu thực vật, cám gạo trích ly và dầu cám thì cần làm những thủ tục gì?
    - Tôi thực hiện nộp hồ sơ thay đổi thông tin đăng ký thuế ở đâu?
    - Khi cơ quan có thẩm quyền tiếp nhận hồ sơ phê duyệt trữ lượng khoáng sản, thời gian giải quyết là bao nhiêu lâu?
    - Việc tiếp nhận, xử lý hồ sơ giải quyết TTHC theo cơ chế một cửa thuộc thẩm quyền của Bộ Tài nguyên và Môi trường được thực hiện theo văn bản nào?
    - Công ty điều chỉnh lại giá trị dòng chảy tối thiểu sau đập thì lập Hồ sơ điều chỉnh, cấp lại hay cấp mới?
    - Có thể thực hiện dịch vụ công trực tuyến cấp 4 tại bộ được không? Lĩnh vực môi trường gồm những thủ tục  nào thực hiện nộp hồ sơ theo dịch vụ công trực tuyến cấp 4?
    - Trong quá trình thực hiện Nghị định 82/2017/NĐ-CP đã gặp vướng mắc khi áp dụng mức thu tiền cấp quyền khai thác tài nguyên nước (M) của các công trình cấp nước sinh hoạt tập trung cho các hộ gia đình, cá nhân ( có thu tiền sử dụng nước). Như vậy, các trường hợp này sẽ áp dụng mức thu tiền cấp quyền khai thác tài nguyên nước (M) bằng bao nhiêu?
    - Trình tự thu hồi đất do người sử dụng đất tự nguyện trả lại đất?
    - Nhà thầu nước ngoài phải đáp ứng điều kiện nào để được cấp giấy phép hoạt động đo đạc và bản đồ?
    - Những hoạt động đo đạc và bản đồ nào cần phải có giấy phép?
    - Chứng chỉ hành nghề đo đạc và bản đồ được cấp đổi trong trường hợp nào?
    - Tổ chức, cá nhân nào có đủ điều kiện tham gia đấu giá quyền khai thác khoáng sản?
    - Theo quy định tại Thông tư số 41/2016/TT-BTTTT ngày 26/12/2016 có quy định về tuổi thiết bị in khi nhập khẩu, nhưng quy định tại Thông tư số 22/2018/TT-BTTTT ngày 28/12/2018 của Bộ trưởng Bộ Thông tin và Truyền thông thì không thấy quy định này. Xin hỏi chúng tôi phải thực hiện theo quy định nào?
    - Đại lý của doanh nghiệp kinh doanh dịch vụ bưu chính thành lập theo pháp luật Việt Nam có phải làm thủ tục đề nghị cấp giấy phép bưu chính, văn bản xác nhận thông báo hoạt động bưu chính hay không?
    - Các trường hợp nào khi nhập khẩu xuất bản phẩm không kinh doanh không phải đề nghị cấp giấy phép?
    - Điều kiện về kỹ thuật như thế nào để cấp Giấy phép thiết lập mạng xã hội?
    - Giấy phép thành lập Văn phòng đại diện nhà xuất bản nước ngoài tại Việt Nam được cấp lại trong những trong hợp nào?
    - Trình tự, thủ tục cấp lại chứng chỉ hành nghề biên tập (đối với trường hợp bị mất hoặc hư hỏng)?
    - Thời hạn giải quyết hồ sơ đối với đài truyền thanh không dây là bao nhiêu ngày kể từ khi nhận được hồ sơ đầy đủ và hợp lệ?
    - Theo quy định pháp luật về bưu chính, hồ sơ đề nghị cấp giấy phép bưu chính cần có: “Giấy chứng nhận đăng ký kinh doanh (ĐKKD) do doanh nghiệp tự đóng dấu xác nhận và chịu trách nhiệm về tính chính xác của bản sao”. Vậy giấy chứng nhận ĐKKD do UBND Phường xác thực có được chấp nhận hay không?
    - Các phương thức nộp phí/lệ phí như thế nào?
    - Thời hạn giải quyết hồ sơ thay đổi nội dung ghi trong Giấy phép xuất bản chuyên trang là bao nhiêu lâu?
    - Tôi sống ở thành phố Cần Thơ. Bố tôi sống ở thành phố Hà Nội. Tôi muốn ủy quyền cho Bố tôi làm thủ tục mua nhà ở tại thành phố Hà Nội. Chúng tôi có bắt buộc phải cùng đến một Văn phòng công chứng để thực hiện công chứng hợp đồng ủy quyền không?
    - Chi nhánh tổ chức hòa giải thương mại nước ngoài tại Việt Nam sau khi được cấp Giấy phép thành lập cần thực hiện thủ tục gì để có thể hoạt động?
    - Tôi muốn lấy vợ cùng xã thì tôi phải làm thế nào, chuẩn bị những giấy tờ gì?
    - Tôi muốn rút yêu cầu trợ giúp pháp lý cần thực hiện những thủ tục gì?
    - Tôi sinh năm 1946, hiện nay, tôi có yêu cầu đăng ký khai sinh. Tuy nhiên, tôi không nhớ rõ ngày, tháng sinh và các giấy tờ của tôi được cấp trước đây chỉ có năm sinh. Tôi muốn hỏi: trường hợp của tôi trong các giấy tờ trước đây không có thông tin ngày, tháng sinh thì có được thêm ngày, tháng sinh trong Giấy khai sinh khi đăng ký khai sinh lại hay không?
    - Thẩm quyền cấp Phiếu lý lịch tư pháp được quy định như thế nào?
    - Đăng ký hoạt động Văn phòng công chứng được chuyển đổi từ Văn phòng công chứng do một công chứng viên thành lập cần những giấy tờ gì?
    - Tôi là nam, năm nay 19 tuổi, muốn lấy vợ, vậy tôi có được đăng ký kết hôn không?
    - Tôi muốn thay đổi tên gọi trong Giấy đăng ký hoạt động của Trung tâm hòa giải thương mại thì cần thực hiện thủ tục gì?
    - Tôi muốn nhận lại tiền tạm ứng án phí thì cần những giấy tờ gì?
    - Để được cấp thẻ nhân viên chăm sóc nạn nhân bạo lực gia đình, tôi phải đến đâu thể thực hiện thủ tục này? Hồ sơ gồm những gì?
    - Nhằm chia sẻ với bà con và giúp mọi người trên địa bàn có điều kiện tiếp cận sách, tác phẩm văn học từ nguồn sách huy động được, chúng tôi muốn thành lập và mở cửa thư viện để phục vụ cộng đồng. Đề nghị cho biết các điều kiện để thành lập thư viện tư nhân hiện được quy định như thế nào?
    - Hồ sơ của thủ tục cấp Giấy phép cho phép tổ chức, cá nhân Việt Nam thuộc địa phương ra nước ngoài biểu diễn nghệ thuật, trình diễn thời trang gồm những gì? Thời gian thực hiện trong bao lâu?
    - Hiện vật của bảo tàng quốc gia để được công nhận là bảo vật quốc gia cần đáp ứng những điều kiện nào?
    - Những mặt hàng nhập khẩu nào thuộc danh mục hàng hóa kiểm tra chuyên ngành của Bộ Văn hóa, Thể thao và Du lịch và theo quy định của văn bản quy phạm pháp luật nào?
    - Đối với khu du lịch nằm trên địa bàn từ 02 đơn vị hành chính cấp tỉnh trở lên thì cơ quan nào có trách nhiệm lập hồ sơ đề nghị công nhận khu du lịch quốc gia và cơ quan nào có thẩm quyền tiếp nhận hồ sơ đề nghị công nhận khu du lịch quốc gia?
    - Hồ sơ đề nghị xét tặng danh hiệu “Nghệ sĩ ưu tú” gồm những loại giấy tờ gì?
    - Mức phí thẩm định của thủ tục cấp phép phổ biến tác phẩm sáng tác trước năm 1975 hoặc tác phẩm của người Việt Nam đang sinh sống và định cư ở nước ngoài như thế nào?
    - Tiêu chuẩn để xét tặng danh hiệu “Giải thưởng Nhà nước” về văn học, nghệ thuật gồm những gì?
    - TTHC nhập khẩu văn hóa phẩm được quy định tại văn bản quy phạm pháp luật nào?
    - Mẫu giấy phép di dời công trình quy định ở Thông tư nào?
    - Chứng nhận hợp quy sản phẩm, hàng hóa VLXD nhập khẩu được thực hiện theo phương thức nào?
    - Đề nghị Bộ Xây dựng hướng dẫn tôi thành phần hồ sơ đề nghị cấp giấy phép xây dựng đối với trường hợp xây dựng mới công trình của các tổ chức quốc tế tại Việt Nam bao gồm những tài liệu gì?
    - Mẫu đơn đề nghị cấp giấy phép chặt hạ, dịch chuyển cây xanh đô thị?
    - Tổ chức muốn nộp hồ sơ đề nghị cấp chứng chỉ năng lực hoạt động xây dựng lần đầu thì cần những tài liệu gì?
    - Khi nào thì chủ đầu tư được thuê tư vấn quản lý dự án và điều kiện năng lực tư vấn quản lý dự án? Nếu được làm tư vấn quản lý dự án mà chưa đủ năng lực thì có được liên danh tư vấn quản lý dự án với một công ty khác đủ năng lực không?
    - Đồ án quy hoạch chung xây dựng xã gồm các nội dung nào?
    - Trách nhiệm của Vụ Vật liệu xây dựng trong hoạt động công bố hợp quy đối với sản phẩm, hàng hóa vật liệu xây dựng?
    - Công trình, nhà ở riêng lẻ được cấp giấy phép xây dựng có thời hạn, khi hết thời hạn mà kế hoạch thực hiện quy hoạch xây dựng chưa được triển khai thì cơ quan đã cấp giấy phép xây dựng có trách nhiệm thông báo cho chủ sở hữu công trình hoặc người được giao sử dụng công trình về điều chỉnh quy hoạch xây dựng, vậy chủ công trình có phải thực hiện gia hạn giấy phép xây dựng có thời hạn không?
    - Điều kiện cấp chứng chỉ hành nghề quản lý dự án hạng II, hạng III là gì?
    - Cơ quan nào có thẩm quyền cấp số tiếp nhận Phiếu công bố sản phẩm mỹ phẩm nhập khẩu?
    - Kho bảo quản đã được cấp GSP cho dược liệu, thuốc cổ truyền có nhu cầu chuyển địa điểm mới thì cần thực hiện thủ tục như thế nào?
    - Đối với tài liệu kỹ thuật theo mẫu 02 quy định tại Phụ lục VIII Nghị định 169/2018/NĐ-CP đã làm bằng tiếng Anh và nộp trước khi có Nghị định 169 thì giải quyết như thế nào
    - Việc thẩm định để cấp, cấp lại, điều chỉnh giấy phép hoạt động đối với cơ sở khám bệnh, chữa bệnh thuộc thẩm quyền của Bộ Y tế được quy định như thế nào?
    - Những đối tượng nào được cấp chứng chỉ hành nghề khám bệnh, chữa bệnh?
    - Cập nhật tờ hướng dẫn sử dụng theonooij dung tờ HDSD của thuốc biệt dược gốc thì có phải nộp hồ đề nghị thay đổi bổ sung Giấy đăng ký lưu hành không?
    - Hồ sơ đề nghị cấp giấy phép hoạt động đối với cơ sở khám bệnh, chữa bệnh bao gồm những giấy tờ gì ?
    - Hiện Công ty có phiếu tiếp nhận đủ điều kiện phân loại trang thiết bị y tế và đã thực hiện phân loại theo mẫu phụ lục Nghị định 36/2016/NĐ-CP. Nhưng theo quy định của Nghị định 169/2018/NĐ-CP sửa đổi Nghị định 36/2016/NĐ-CP: mẫu bảng phân loại đã thay đổi, vậy Công ty có được sử dụng bảng cũ không? Hay phải làm lại theo quy định mới của Nghị định 169/2018/NĐ-CP.
    - Tham khảo các văn bản liên quan đến cấp giấy phép hoạt động ngân hàng mô thì tìm ở đâu? hay liên hệ với ai để có tài liệu tham khảo?
    - Công ty tiến hành nhập khẩu hóa chất, chất thử chuẩn được Cục Quản lý Dược và Thực phẩm Mỹ phân loại sử dụng trong phòng thí nghiệm cung cấp cho Bệnh Viện để nghiên cứu (không dùng cho mục đích khám chữa bệnh). Hỏi những hóa chất trên có phải xin giấy phép nhập khẩu của Bộ Y tế không?
    - Khi chứng thư số sắp hết hạn cần xử lý như thế nào để gia hạn chứng thư số.
    - Trình tự thực hiện thủ tục chấp thuận mua bán, chuyển nhượng phần vốn góp tại ngân hàng thương mại trách nhiệm hữu hạn hai thành viên trở lên như thế nào?
    - Cá nhân nghỉ từ 40 ngày làm việc trở lên có được xét tặng danh hiệu “Lao động tiên tiến” không?
    - Trong thời gian bao lâu kể từ ngày Ngân hàng Nhà nước ra quyết định sửa đổi, bổ sung Giấy phép đối với việc thay đổi địa điểm đặt trụ sở chính, tổ chức tài chính vi mô phải hoạt động tại địa điểm mới?
    - Hồ sơ đề nghị cấp tín dụng vượt giới hạn gồm những giấy tờ gì?
    - Hồ sơ đề nghị giám định tiền không đủ tiêu chuẩn lưu thông?
    - Quy trình tạo và gửi một hồ sơ TTHC trên dịch vụ công gồm những bước nào?
    - Hồ sơ đề nghị chấp thuận thay đổi địa chỉ (không thay đổi địa điểm) đặt trụ sở chính của quỹ tín dụng nhân dân gồm những gì?
    - Hồ sơ đề nghị xét tặng danh hiệu “Chiến sĩ thi đua toàn quốc” gồm giấy tờ gì?
    - Hồ sơ đề nghị chấp thuận thay thời hạn hoạt động của tổ chức tài chính vi mô gồm những gì?
    - Xin hỏi giữa khiếu nại và tố cáo khác nhau như thế nào?
    - Nội dung nào phải công khai, minh bạch?
    - Trách nhiệm của cơ quan, tổ chức, đơn vị và doanh nghiệp, tổ chức khu vực ngoài nhà nước trong phòng, chống tham nhũng?
    - Xin hỏi, người dân có thể lên Trụ sở tiếp công dân ở Trung ương để khiếu nại, tố cáo, phản ánh những vấn đề ở địa phương không?
    - Trách nhiệm của người có quyền, lợi ích hợp pháp liên quan trong việc thi hành quyết định giải quyết khiếu nại có hiệu lực pháp luật?
    - Đối thoại trong khiếu nại quyết định kỷ luật cán bộ, công chức được tổ chức như thế nào?
    - Tại sao phải đối thoại, giải quyết lần nào phải tổ chức đối thoại?
    - Đối với tố cáo có dấu hiệu tội phạm thì cơ quan, tổ chức, cá nhân tiếp nhận tố cáo cần phải xử lý như thế nào?
    - Phương thức và thời điểm kê khai tài sản, thu nhập thực hiện như thế nào?
    - Người tố cáo có các quyền như thế nào?
    '''
    
    print(convert_to_no_accents(a))